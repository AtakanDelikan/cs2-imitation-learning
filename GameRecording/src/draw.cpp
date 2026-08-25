#include "draw.hpp"
#include <windows.h>
#include <vector>
#include <mutex>

namespace
{
    struct Box {
        int x, y;
    };

    std::vector<Box> g_boxes;
    std::vector<Box> r_boxes;
    std::mutex g_mutex;
    HWND g_hwnd = nullptr;

    const COLORREF TRANSPARENT_COLOR = RGB(255, 0, 255);
    const COLORREF RED_COLOR = RGB(255, 0, 0);
    // const COLORREF BOX_COLOR = RGB(0, 247, 255);
    const COLORREF BOX_COLOR = RGB(0, 0, 0);
    const char* CLASS_NAME = "OverlayWindowClass"; // Standard char*

    LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        switch (uMsg)
        {
        case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            std::lock_guard<std::mutex> lock(g_mutex);
            HBRUSH boxBrush = CreateSolidBrush(BOX_COLOR);
            HBRUSH redBrush = CreateSolidBrush(RED_COLOR);
            for (const auto& box : g_boxes)
            {
                RECT rect = { box.x - 2, box.y - 2, box.x + 2, box.y + 2 };
                FillRect(hdc, &rect, boxBrush);
            }
            for (const auto& box : r_boxes)
            {
                RECT rect = { box.x - 2, box.y - 2, box.x + 2, box.y + 2 };
                FillRect(hdc, &rect, redBrush);
            }
            DeleteObject(boxBrush);
            DeleteObject(redBrush);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_ERASEBKGND:
        {
            HDC hdc = (HDC)wParam;
            RECT rect;
            GetClientRect(hwnd, &rect);
            HBRUSH bgBrush = CreateSolidBrush(TRANSPARENT_COLOR);
            FillRect(hdc, &rect, bgBrush);
            DeleteObject(bgBrush);
            return 1;
        }
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

namespace draw
{
    bool init()
    {
        if (g_hwnd) return true;

        // Use WNDCLASSA (The 'A' stands for ANSI/char)
        WNDCLASSA wc = { 0 };
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = CLASS_NAME;

        // Use RegisterClassA
        if (!RegisterClassA(&wc)) return false;

        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);

        // Use CreateWindowExA
        g_hwnd = CreateWindowExA(
            WS_EX_TOPMOST | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW,
            CLASS_NAME,
            "Overlay Window",
            WS_POPUP | WS_VISIBLE,
            0, 0, screenWidth, screenHeight,
            nullptr, nullptr, wc.hInstance, nullptr
        );

        if (!g_hwnd) return false;

        SetLayeredWindowAttributes(g_hwnd, TRANSPARENT_COLOR, 0, LWA_COLORKEY);
        return true;
    }

    void add_box(int x, int y)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_boxes.push_back({ x, y });
        if (g_hwnd) InvalidateRect(g_hwnd, nullptr, TRUE);
    }

    void add_box_red(int x, int y)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        r_boxes.push_back({ x, y });
        if (g_hwnd) InvalidateRect(g_hwnd, nullptr, TRUE);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_boxes.clear();
        r_boxes.clear();
        if (g_hwnd) InvalidateRect(g_hwnd, nullptr, TRUE);
    }

    void render()
    {
        if (!g_hwnd) { if (!init()) return; }
        MSG msg = { 0 };
        while (GetMessage(&msg, nullptr, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
}