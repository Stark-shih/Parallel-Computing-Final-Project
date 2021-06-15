#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include "solver.h"
using namespace sf;

void draw_mouse(Uint8 * pixels, int W, int mouse_x, int mouse_y) {
    //int index = (mouse_x+ mouse_y*W) * 4;
    //pixels[index] = 255;
    //pixels[index+1] = 255;
    //pixels[index+2] = 255;
    int x = std::max(mouse_x - 20, 0);
    int y = std::max(mouse_y - 20, 0);
    int x_m = std::min(mouse_x + 20, 800);
    int y_m = std::min(mouse_y + 20, 600);
    for (register int i = x; i < x_m; i++) {
        for (register int j = y; j < y_m; j++) {
            pixels[(i + j * W) * 4] = 255;
            pixels[(i + j * W) * 4+1] = 255;
            pixels[(i + j * W) * 4+2] = 255;
        }
    }
}

int main()
{
    int W = 512, H = 512;
    RenderWindow window(VideoMode(W, H), "test");
    //window.setFramerateLimit(60);

    Uint8 * pixels = new Uint8[W*H*4];
    Texture texture;
    texture.create(W, H);
    Sprite sprite(texture);
    for (register int i = 0; i < W * H * 4; i += 4) {
        pixels[i] = 0;
        pixels[i + 1] = 0;
        pixels[i + 2] = 0;
        pixels[i + 3] = 255;
    }

    Clock clock;
    Time t;
    Vector2i mouse_pos;
    bool move_flag = false;
    while (window.isOpen())
    {
        //Vector2i localPosition = sf::Mouse::getPosition(window);
        Event event;
        while (window.pollEvent(event))
        {
            switch (event.type) {
                case Event::Closed:
                    window.close();
                    break;
                case Event::MouseMoved:
                    move_flag = true;
                    mouse_pos.x = event.mouseMove.x;
                    mouse_pos.y = event.mouseMove.y;
                    break;
                default:
                    break;
            }

        }

        for (register int i = 0; i < W * H * 4; i += 4) {
            if (pixels[i] > 0) {
                pixels[i] = std::max((int)pixels[i]-1, 0);
                pixels[i + 1] = std::max((int)pixels[i]-1, 0);
                pixels[i + 2] = std::max((int)pixels[i]-1, 0);
            }
            //pixels[i + 3];
        }
        if(move_flag)
            draw_mouse(pixels, W, mouse_pos.x, mouse_pos.y);
        move_flag = false;
        texture.update(pixels);
        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
