#pragma once
#include <windows.h>
#include <atomic>

void handle_recording(HANDLE driver, std::uintptr_t client, std::uintptr_t server);