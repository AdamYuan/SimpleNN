#include "Loader.hpp"
#include <fstream>

static inline int byte_to_int32(unsigned char *byte4)
{ return (byte4[0]<<24) | (byte4[1]<<16) | (byte4[2]<<8) | byte4[3]; }

MnistLoader::MnistLoader(const char *file1, const char *file2)
{
	Load(file1, file2);
}

void MnistLoader::Load(const char *file1, const char *file2)
{
	std::ifstream in1{file1, std::ios::binary}, in2{file2, std::ios::binary};
	std::vector<unsigned char> image_buffer
	{std::istreambuf_iterator<char>(in1), std::istreambuf_iterator<char>()};
	std::vector<unsigned char> label_buffer
	{std::istreambuf_iterator<char>(in2), std::istreambuf_iterator<char>()};

	if(byte_to_int32(image_buffer.data()) == 2049)
		std::swap(image_buffer, label_buffer);

	int num_item = byte_to_int32(image_buffer.data() + 4);
	if(num_item != byte_to_int32(label_buffer.data() + 4))
		printf("MNIST file item count doesn't match\n");

	int rows = byte_to_int32(image_buffer.data() + 8);
	int cols = byte_to_int32(image_buffer.data() + 12);
	int imgsz = rows*cols;

	unsigned char *images = image_buffer.data() + 16;
	unsigned char *labels = label_buffer.data() + 8;

	m_data_set.resize(num_item);

	for(int i = 0; i < num_item; ++i)
	{
		m_data_set[i].m_inputs.resize(imgsz);
		m_data_set[i].m_expected.resize(10, 0.0);

		for(int j = 0; j < imgsz; ++j)
		{
			int l = images[i*imgsz + j];
			m_data_set[i].m_inputs[j] = float(l) / 255.0f;
		}
		m_data_set[i].m_expected[int(labels[i])] = 1.0;
	}
	printf("MNIST data loaded from %s and %s\n", file1, file2);
}
