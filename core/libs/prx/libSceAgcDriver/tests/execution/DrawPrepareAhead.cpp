#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawPipeline;
using Redo = DrawPipeline::PrepareAheadRedo;

constexpr std::byte Kept{0x40};
constexpr std::uint32_t BaseVertexLocation = 0x8c;
constexpr std::size_t IndexBlockBytes = 65536;

alignas(256) constexpr std::array<std::uint32_t, 20> VertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060208u, 0x7e0802f2u, 0x7e0a02ffu, 0x44800000u, 0xf80000d1u, 0x00000005u, 0xf80008cfu, 0x04030201u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
alignas(256) std::array<std::byte, 1024> DirectMemory{};
alignas(256) std::array<std::byte, 1024> IndexedMemory{};
alignas(256) std::array<std::byte, 1024> CpuWindowMemory{};
alignas(256) std::array<std::byte, 1024> AdoptedMemory{};
alignas(256) std::array<std::byte, 1024> LabelWindowMemory{};
alignas(256) std::array<std::byte, 1024> RegistryMemory{};
alignas(4096) std::array<std::byte, 4096> SpareMemory{};
std::uint16_t* Indices = nullptr;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header PixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void AllocateIndices() {
#ifdef _WIN32
    auto* block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(IndexBlockBytes, IndexBlockBytes));
    Require(block != nullptr, "cannot reserve the index block in the guest arena");
    GuestArena::GuestArenaCommit_nid_postfix(block, IndexBlockBytes, PAGE_READWRITE, IndexBlockBytes);
#else
    void* mapped = mmap(nullptr, IndexBlockBytes * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(mapped != MAP_FAILED, "cannot map the index block");
    const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
    auto* block = reinterpret_cast<std::uint8_t*>((begin + IndexBlockBytes - 1) & ~(static_cast<std::uintptr_t>(IndexBlockBytes) - 1));
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, IndexBlockBytes);
#endif
    std::memset(block, 0, IndexBlockBytes);
    GuestAllocations::Mutation().Add(block, IndexBlockBytes, true, true, true);
    Require(AgcDriver::GuestMemory::Watched(reinterpret_cast<std::uint64_t>(block), IndexBlockBytes), "the index block is not write-watched");
    Indices = reinterpret_cast<std::uint16_t*>(block);
}

void SetIndices(std::uint16_t first, std::uint16_t second, std::uint16_t third) {
    Indices[0] = first;
    Indices[1] = second;
    Indices[2] = third;
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void AppendState(std::vector<std::uint32_t>& commands) {
    const auto vertex = reinterpret_cast<std::uintptr_t>(VertexCode.data());
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0x01200000}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 1u << 1u},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 1u << 1u}, {0x00c, std::bit_cast<std::uint32_t>(1.0f)},
    };
    Append(commands, 0x79, {0x242, 4});
    Append(commands, 0x79, {0x24a, 0});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
}

void AppendTarget(std::vector<std::uint32_t>& commands, std::array<std::byte, 1024>& memory) {
    const auto color = reinterpret_cast<std::uintptr_t>(memory.data());
    Append(commands, 0x69, {0x318, static_cast<std::uint32_t>(color >> 8u)});
    Append(commands, 0x69, {0x390, static_cast<std::uint32_t>(color >> 40u)});
    Append(commands, 0x76, {BaseVertexLocation, 0});
}

void AppendIndexedDraw(std::vector<std::uint32_t>& commands) {
    const auto address = reinterpret_cast<std::uintptr_t>(Indices);
    Append(commands, 0x2a, {0});
    Append(commands, 0x27, {3, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 3, 0});
}

void AppendLabel(std::vector<std::uint32_t>& commands, const void* target, std::uint32_t value) {
    const auto address = reinterpret_cast<std::uintptr_t>(target);
    commands.insert(commands.end(), {0xc0064900u, 0x514u, (1u << 29u) | (2u << 24u), static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), value, 0, 0});
}

std::size_t Covered(const std::array<std::byte, 1024>& memory) {
    return static_cast<std::size_t>(std::count_if(memory.begin(), memory.end(), [](std::byte value) { return value != Kept; }));
}

void SameAsDirect(const std::array<std::byte, 1024>& memory, const std::string& what) {
    for (std::size_t index = 0; index < DirectMemory.size(); ++index) {
        Require(memory[index] == DirectMemory[index], what + ": byte " + std::to_string(index) + " is " + std::to_string(static_cast<unsigned>(memory[index])) + ", the direct draw wrote " + std::to_string(static_cast<unsigned>(DirectMemory[index])));
    }
}

void Flush(const std::array<std::byte, 1024>& memory) {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(memory.data()), memory.size());
}

struct Blocker {
    struct State {
        std::atomic<bool> release{false};
        std::atomic<bool> done{false};
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    void Release() { state->release = true; }
    void Enqueue() {
        DrawPipeline::Queue0().Enqueue([state = state] {
            while (!state->release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            { std::lock_guard gpuLock(AgcDriver::GuestMemory::GpuMutex()); }
            state->done = true;
        }, {});
    }
    void Join() const {
        const auto start = std::chrono::steady_clock::now();
        while (!state->done.load()) {
            Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(20), "the blocker never ran");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

void Finish() {
    auto idle = std::async(std::launch::async, [] { AgcDriverWaitIdle_nid_postfix(); });
    if (idle.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
        std::fputs("draw prepare ahead: the queue worker and the draw pipeline deadlocked\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    }
    idle.get();
}

void SubmitBehindTheBlocker(Blocker& blocker, std::vector<std::uint32_t>& commands, std::size_t queued) {
    blocker.Enqueue();
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    const auto start = std::chrono::steady_clock::now();
    while (DrawPipeline::Queue0().Outstanding() < queued) {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
            const auto outstanding = DrawPipeline::Queue0().Outstanding();
            blocker.Release();
            Finish();
            Require(false, "the draw did not queue behind the blocker (" + std::to_string(outstanding) + " items outstanding, expected " + std::to_string(queued) + ")");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::string Describe(const DrawPipeline::PrepareAheadCounts& before, const DrawPipeline::PrepareAheadCounts& after) {
    std::string text = " prepared " + std::to_string(after.prepared - before.prepared) + ", adopted " + std::to_string(after.adopted - before.adopted) + ", refused " + std::to_string(after.refused - before.refused) + ", redone:";
    for (std::size_t reason = 0; reason < after.redone.size(); ++reason) text += " " + std::string(DrawPipeline::PrepareAheadRedoName(static_cast<Redo>(reason))) + " " + std::to_string(after.redone[reason] - before.redone[reason]);
    return text;
}

void ExpectCounts(const DrawPipeline::PrepareAheadCounts& before, const DrawPipeline::PrepareAheadCounts& after, std::uint64_t prepared, std::uint64_t adopted, std::optional<Redo> redone, const std::string& what) {
    Require(after.prepared - before.prepared == prepared, what + ": the draw was not prepared ahead as expected:" + Describe(before, after));
    Require(after.adopted - before.adopted == adopted, what + ": the prepared draw was not adopted as expected:" + Describe(before, after));
    for (std::size_t reason = 0; reason < after.redone.size(); ++reason) {
        const std::uint64_t expected = redone.has_value() && static_cast<std::size_t>(*redone) == reason ? 1u : 0u;
        Require(after.redone[reason] - before.redone[reason] == expected, what + ": the prepared draw was not redone as expected:" + Describe(before, after));
    }
}

void DrawDirectReference() {
    DirectMemory.fill(Kept);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendTarget(commands, DirectMemory);
    Append(commands, 0x2d, {3, 2});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the direct draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    Flush(DirectMemory);
    const auto direct = Covered(DirectMemory);
    Require(direct != 0 && direct != DirectMemory.size(), "the direct draw covered " + std::to_string(direct) + " of " + std::to_string(DirectMemory.size()) + " bytes, expected an edge through the target");
}

void IndexedDrawMatchesTheDirectOne() {
    IndexedMemory.fill(Kept);
    SetIndices(0, 1, 2);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendTarget(commands, IndexedMemory);
    AppendIndexedDraw(commands);
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the indexed draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    Flush(IndexedMemory);
    SameAsDirect(IndexedMemory, "the indexed draw of the three vertices");
}

void CpuWriteInTheWindow() {
    CpuWindowMemory.fill(Kept);
    SetIndices(0, 0, 2);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendTarget(commands, CpuWindowMemory);
    AppendIndexedDraw(commands);
    const auto before = DrawPipeline::PrepareAhead();
    Blocker blocker;
    SubmitBehindTheBlocker(blocker, commands, 2);
    SetIndices(0, 1, 2);
    blocker.Release();
    Finish();
    blocker.Join();
    Flush(CpuWindowMemory);
    const auto after = DrawPipeline::PrepareAhead();
    SameAsDirect(CpuWindowMemory, "the draw whose indices the CPU wrote while it was queued");
    ExpectCounts(before, after, 1, 0, Redo::CpuWrite, "a CPU write into the index range while the draw was queued");
}

void PreparedDrawIsAdopted() {
    AdoptedMemory.fill(Kept);
    SetIndices(0, 1, 2);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendTarget(commands, AdoptedMemory);
    AppendIndexedDraw(commands);
    const auto before = DrawPipeline::PrepareAhead();
    Blocker blocker;
    SubmitBehindTheBlocker(blocker, commands, 2);
    blocker.Release();
    Finish();
    blocker.Join();
    Flush(AdoptedMemory);
    const auto after = DrawPipeline::PrepareAhead();
    SameAsDirect(AdoptedMemory, "the draw prepared ahead and adopted");
    ExpectCounts(before, after, 1, 1, std::nullopt, "a draw whose inputs nobody wrote while it was queued");
}

void RegistryChangeInTheWindow() {
    RegistryMemory.fill(Kept);
    SetIndices(0, 1, 2);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendTarget(commands, RegistryMemory);
    AppendIndexedDraw(commands);
    const auto before = DrawPipeline::PrepareAhead();
    Blocker blocker;
    SubmitBehindTheBlocker(blocker, commands, 2);
    GuestAllocations::Mutation().Add(SpareMemory.data(), SpareMemory.size(), true, true, true);
    blocker.Release();
    Finish();
    blocker.Join();
    GuestAllocations::Mutation().Remove(SpareMemory.data());
    Flush(RegistryMemory);
    const auto after = DrawPipeline::PrepareAhead();
    SameAsDirect(RegistryMemory, "the draw queued across a registry change");
    ExpectCounts(before, after, 1, 0, Redo::Registry, "a registry change while the draw was queued");
}

void QueuedLabelWriteInTheWindow() {
    LabelWindowMemory.fill(Kept);
    SetIndices(0, 0, 2);
    std::vector<std::uint32_t> commands;
    AppendState(commands);
    AppendLabel(commands, Indices, 0x00010000u);
    AppendTarget(commands, LabelWindowMemory);
    AppendIndexedDraw(commands);
    const auto before = DrawPipeline::PrepareAhead();
    Blocker blocker;
    SubmitBehindTheBlocker(blocker, commands, 3);
    blocker.Release();
    Finish();
    blocker.Join();
    Flush(LabelWindowMemory);
    const auto after = DrawPipeline::PrepareAhead();
    Require(Indices[1] == 1, "the queued label did not write the second index");
    SameAsDirect(LabelWindowMemory, "the draw whose indices a queued label wrote");
    ExpectCounts(before, after, 1, 0, Redo::QueuedWrite, "a queued write over the index range");
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
    _putenv_s("APS5_DRAW_PREPARE_AHEAD", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
    setenv("APS5_DRAW_PREPARE_AHEAD", "1", 1);
#endif
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        AllocateIndices();
        DrawDirectReference();
        IndexedDrawMatchesTheDirectOne();
        CpuWriteInTheWindow();
        PreparedDrawIsAdopted();
        RegistryChangeInTheWindow();
        QueuedLabelWriteInTheWindow();
        AgcDriverShutdown_nid_postfix();
        std::puts("draw prepare ahead tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
