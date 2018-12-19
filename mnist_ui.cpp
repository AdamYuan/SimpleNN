#include <iostream>
#include <algorithm>
#include "NN/NN.hpp"
#include "MNIST/Util.hpp"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

constexpr int kGridSize = 14, kSize = 28*kGridSize;
constexpr float kMinRadius = 8.0, kMaxRadius = 24.0, kRadiusStep = 1.0;
sf::RenderWindow window;
sf::RenderTexture render_tex;
sf::Event event;
float radius{10.0f};
sf::CircleShape brush, cursor;
SimpleNN snn;

void InitWindow()
{
	window.create(sf::VideoMode(kSize, kSize), "Mnist Demo", sf::Style::Titlebar | sf::Style::Close);
	render_tex.create(kSize, kSize);

	brush.setFillColor(sf::Color(0, 0, 0));
	cursor.setFillColor(sf::Color(0, 0, 0, 100));
	brush.setRadius(radius);
	cursor.setRadius(radius);
}
void Paint()
{
	sf::Vector2i xy = sf::Mouse::getPosition(window);
	if(xy.x >= 0 && xy.y < kSize && xy.y >= 0 && xy.y < kSize)
	{
		int x = std::max(0, std::min(xy.x, kSize)) - radius, y = kSize - std::max(0, std::min(xy.y, kSize)) - radius;
		brush.setPosition(x, y);
		render_tex.draw(brush);
	}
}
void Clear()
{
	render_tex.clear(sf::Color(255, 255, 255));
}
void Cursor()
{
	sf::Vector2i xy = sf::Mouse::getPosition(window);
	if(xy.x >= 0 && xy.y < kSize && xy.y >= 0 && xy.y < kSize)
	{
		int x = std::max(0, std::min(xy.x, kSize)) - radius, y = std::max(0, std::min(xy.y, kSize)) - radius;
		cursor.setPosition(x, y);
		window.draw(cursor);
	}
}
void Recognize()
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

	InitWindow();
	Clear();

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
				Recognize();
				Clear();
			}
			if(event.type == sf::Event::EventType::MouseButtonPressed)
				mouse_down = true;
			if(event.type == sf::Event::EventType::MouseButtonReleased)
				mouse_down = false;
			if(event.type == sf::Event::EventType::MouseWheelScrolled)
			{
				radius += kRadiusStep * (event.mouseWheel.x > 0 ? -1 : 1);
				radius = std::min(std::max(kMinRadius, radius), kMaxRadius);
				brush.setRadius(radius);
				cursor.setRadius(radius);
			}
		}
		if(mouse_down)
			Paint();

		window.draw(sf::Sprite(render_tex.getTexture()));
		Cursor();
		window.display();
	}

	return EXIT_SUCCESS;
}
