#include "memory.hpp"
#include "client_dll.hpp"
#include "server_dll.hpp"
#include "offsets.hpp"
#include "features.hpp"
#include "draw.hpp"
#include <iostream>
#include <atomic>
#include <future>
#include <thread>

int main() {

    const DWORD pid = get_process_id(L"cs2.exe");
    if (pid == 0) {
        std::cout << "Failed to find cs2.\n";
        std::cin.get();
        return 1;
    }
    const HANDLE driver = CreateFileW(L"\\\\.\\xdriver", GENERIC_READ, 0, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (driver == INVALID_HANDLE_VALUE) {
        std::cout << "Failed to create our driver handle.\n";
        std::cin.get();
        return 1;
    }
    if (driver::attach_to_process(driver, pid) == true) {
        std::cout << "Attachment successful.\n";
        if (const std::uintptr_t client = get_module_base(pid, L"client.dll"); client != 0) {
            std::cout << "Client found.\n";
            if (const std::uintptr_t server = get_module_base(pid, L"server.dll"); server != 0) {
                auto recordingFuture = std::async(std::launch::async, handle_recording, driver, client, server);
                recordingFuture.get();
            }
        }
    }
    CloseHandle(driver);
    std::cin.get();
    return 0;
}


