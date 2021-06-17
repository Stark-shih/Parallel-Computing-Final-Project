#include <SFML/Graphics.hpp>
#include <iostream>
#include <cmath>
#include "solver.h"
using namespace sf;

int main()
{
    int W = 200, H = 200;
    RenderWindow window(VideoMode(W, H), "test");
    //window.setFramerateLimit(60);

    Uint8* pixels = new Uint8[W * H * 4];
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
    Vector2i new_pos, old_pos;
    float2 forceVector = make_float2(0, 0);
    float2 forceOrigin = make_float2(0, 0);
    bool click_flag = false;
    Solver stableSolver(W, H, W);
    stableSolver.reset();
    float  timestep = 0.1;
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
            case Event::Event::MouseButtonReleased:
                click_flag = false;
                //mouse_pos.x = event.mouseButton.x;
                //mouse_pos.y = event.mouseButton.y;
                //std::cout << "moouse up " << mouse_pos.x << ", " << mouse_pos.y << "\n";
                break;
            case Event::MouseButtonPressed:
                click_flag = true;      
                old_pos.x = event.mouseButton.x;
                old_pos.y = event.mouseButton.y;
                new_pos.x = event.mouseButton.x;
                new_pos.y = event.mouseButton.y;
                //std::cout << "moouse down " << old_pos.x << ", " << old_pos.y << "\n";
                break;
            case Event::MouseMoved:
                if (click_flag) {
                    new_pos.x = event.mouseMove.x;
                    new_pos.y = event.mouseMove.y;
                    std::cout << "moouse down " << old_pos.x << ", " << old_pos.y << "\n";
                }
                break;
            default:
                break;
            }

        }


        float elapsed = clock.getElapsedTime().asSeconds();
        if(elapsed > timestep) {
            if (click_flag) {
                forceOrigin = make_float2(old_pos.x, old_pos.y);
                forceVector = make_float2(new_pos.x - old_pos.x, new_pos.y - old_pos.y);
                old_pos = new_pos;
            }
            stableSolver.update(timestep, forceOrigin, forceVector, pixels);
            forceOrigin = make_float2(0, 0);
            forceVector = make_float2(0, 0);
            clock.restart();
        }


        texture.update(pixels);
        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
