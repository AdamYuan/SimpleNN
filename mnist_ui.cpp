#include <iostream>
#include <algorithm>
#include "NN/NN.hpp"
#include "MNIST/Util.hpp"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

constexpr int kGridSize = 20, kSize = 28*kGridSize, kOutSize = kSize / 10;
constexpr float kMinRadius = 8.0, kMaxRadius = 30.0, kRadiusStep = 1.0;
sf::RenderWindow window;
sf::RenderTexture paint_tex, input_tex, output_tex, output_digits_tex;
sf::Event event;
float radius{10.0f};
sf::CircleShape brush_circle, cursor_circle;
sf::RectangleShape input_rect, output_rect;
SimpleNN snn;

void InitWindow()
{
	window.create(sf::VideoMode(kSize*2 + kOutSize, kSize), "Mnist Demo", sf::Style::Titlebar | sf::Style::Close);
	paint_tex.create(kSize, kSize);
	input_tex.create(kSize, kSize);
	output_tex.create(kOutSize, kSize);
	output_digits_tex.create(kOutSize, kSize);

	sf::Font font; font.loadFromFile("./ui/VCR_OSD_MONO_1.001.ttf");
	sf::Text text; 
	text.setFont(font); text.setCharacterSize(kOutSize);
	text.setFillColor(sf::Color(0, 0, 0, 255));
	for(unsigned i = 0; i < 10; ++i)
	{
		text.setPosition(0, i * kOutSize);
		text.setString(std::to_string(i));
		output_digits_tex.draw(text);
	}
	output_digits_tex.display();

	brush_circle.setFillColor(sf::Color(0, 0, 0));
	cursor_circle.setFillColor(sf::Color(0, 0, 0, 100));
	brush_circle.setRadius(radius);
	cursor_circle.setRadius(radius);

	input_rect.setSize(sf::Vector2f(kGridSize, kGridSize));
	output_rect.setSize(sf::Vector2f(kOutSize, kOutSize));
}
void Paint()
{
	sf::Vector2i xy = sf::Mouse::getPosition(window);
	if(xy.x >= 0 && xy.y < kSize && xy.y >= 0 && xy.y < kSize)
	{
		int x = std::max(0, std::min(xy.x, kSize)) - radius, y = std::max(0, std::min(xy.y, kSize)) - radius;
		brush_circle.setPosition(x, y);
		paint_tex.draw(brush_circle);
	}
	paint_tex.display();
}
void Clear()
{
	paint_tex.clear(sf::Color(255, 255, 255));
}
void Cursor()
{
	sf::Vector2i xy = sf::Mouse::getPosition(window);
	if(xy.x >= 0 && xy.y < kSize && xy.y >= 0 && xy.y < kSize)
	{
		int x = std::max(0, std::min(xy.x, kSize)) - radius, y = std::max(0, std::min(xy.y, kSize)) - radius;
		cursor_circle.setPosition(x, y);
		window.draw(cursor_circle);
	}
}
unsigned Recognize()
{
	sf::Image img{paint_tex.getTexture().copyToImage()};
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
		unsigned gx = i % 28, gy = i / 28, c = 255 * nn_input[i];
		input_rect.setPosition(gx * kGridSize, gy * kGridSize);
		input_rect.setFillColor(sf::Color(c, c, c, 255));
		input_tex.draw(input_rect);
		//putchar(nn_input[i] >= 0.25 ? (nn_input[i] >= 0.5 ? (nn_input[i] >= 0.75 ? '@' : '?') : '.') : ' ');
		//if(i % 28 == 27) putchar('\n');
	}
	input_tex.display();
	snn.Evaluate(nn_input);
	unsigned res = std::max_element(snn.GetOutput(), snn.GetOutput() + 10) - snn.GetOutput();
	for(unsigned i = 0; i < 10; ++i)
	{
		unsigned c = 255 * snn.GetOutput()[i];
		output_rect.setPosition(0, i * kOutSize);
		output_rect.setFillColor(sf::Color(c, c, c, 255));
		output_tex.draw(output_rect);
	}
	output_tex.display();
	return res;
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
				window.setTitle("Recognize: " + std::to_string(Recognize()));
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
				brush_circle.setRadius(radius);
				cursor_circle.setRadius(radius);
			}
		}
		if(mouse_down)
			Paint();

		window.draw(sf::Sprite(paint_tex.getTexture()));

		sf::Sprite input_sprite{input_tex.getTexture()};
		input_sprite.setPosition(kSize, 0);
		window.draw(input_sprite);

		sf::Sprite output_sprite{output_tex.getTexture()};
		output_sprite.setPosition(kSize*2, 0);
		window.draw(output_sprite);

		sf::Sprite output_digits_sprite{output_digits_tex.getTexture()};
		output_digits_sprite.setPosition(kSize*2, 0);
		window.draw(output_digits_sprite);
		Cursor();
		window.display();
	}

	return EXIT_SUCCESS;
}
