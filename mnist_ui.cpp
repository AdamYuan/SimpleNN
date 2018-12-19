#include <iostream>
#include <algorithm>
#include "NN/NN.hpp"
#include "MNIST/Util.hpp"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

constexpr int kGridSize = 14, kSize = 28*kGridSize;
constexpr float kBrushRadius = 10.0f;
sf::RenderWindow window;
sf::RenderTexture render_tex;
sf::Event event;
sf::CircleShape brush;
SimpleNN snn;

void init_window()
{
	window.create(sf::VideoMode(kSize, kSize), "Mnist Demo", sf::Style::Titlebar | sf::Style::Close);
	render_tex.create(kSize, kSize);

	brush.setFillColor(sf::Color(0, 0, 0));
	brush.setRadius(kBrushRadius);
}
void paint()
{
	sf::Vector2i xy = sf::Mouse::getPosition(window);
	if(xy.x >= 0 && xy.y < kSize && xy.y >= 0 && xy.y < kSize)
	{
		int x = std::max(0, std::min(xy.x, kSize)) - kBrushRadius, y = kSize - std::max(0, std::min(xy.y, kSize)) - kBrushRadius;
		brush.setPosition(x, y);
		render_tex.draw(brush);
	}
}
void clear()
{
	render_tex.clear(sf::Color(255, 255, 255));
}
void recognize()
{
	sf::Image img{render_tex.getTexture().copyToImage()};
	const sf::Uint8 *ptr = img.getPixelsPtr();
	std::vector<float> nn_input(784);
	for(unsigned i = 0; i < 784; ++i)
	{
		float v = 0.0;
		unsigned gx = i % 28, gy = i / 28;
		unsigned px = gx * (kGridSize << 2), py = gy * kGridSize;
		for(unsigned y = py; y < py + kGridSize; ++y)
			for(unsigned x = px; x < px + (kGridSize << 2); x += 4)
				v += float(ptr[y * (kSize << 2) + x] == 0);
		nn_input[i] = v / float(kGridSize * kGridSize);
	}
	width_normalize(&nn_input);
	for(unsigned i = 0; i < 784; ++i)
	{
		putchar(nn_input[i] >= 0.25 ? (nn_input[i] >= 0.5 ? (nn_input[i] >= 0.75 ? '@' : '?') : '.') : ' ');
		if(i % 28 == 27) putchar('\n');
	}
	snn.Evaluate(nn_input);
	unsigned res = std::max_element(snn.GetOutput(), snn.GetOutput() + 10) - snn.GetOutput();
	printf("recognize: %d\nnn output: ", res);
	for(unsigned i = 0; i < 10; ++i)
		printf("%f ", snn.GetOutput()[i]);
	printf("\n");
}

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: ./MnistUI [snn filename]\n");
		return EXIT_FAILURE;
	}

	snn.Load(argv[1]);

	init_window();
	clear();

	bool mouse_down = false;
	while(window.isOpen())
	{
		while(window.pollEvent(event))
		{
			if(event.type == sf::Event::EventType::Closed)
				window.close();
			if(event.type == sf::Event::EventType::KeyReleased 
					&& event.key.code == sf::Keyboard::Space)
			{
				recognize();
				clear();
			}
			if(event.type == sf::Event::EventType::MouseButtonPressed)
				mouse_down = true;
			if(event.type == sf::Event::EventType::MouseButtonReleased)
				mouse_down = false;
		}
		if(mouse_down)
			paint();

		window.draw(sf::Sprite(render_tex.getTexture()));
		window.display();
	}

	return EXIT_SUCCESS;
}
