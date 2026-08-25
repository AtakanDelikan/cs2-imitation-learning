#pragma once

namespace draw
{
    // Creates the transparent overlay window
    bool init();

    // Add a box (centered at x,y) 10x10 pixels
    void add_box(int x, int y);

    void add_box_red(int x, int y);

    // Clear all boxes
    void clear();

    // Start rendering loop (blocking)
    void render();
}