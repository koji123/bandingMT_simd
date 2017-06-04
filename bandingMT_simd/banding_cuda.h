
#define ENABLE_PERF

#include <Windows.h>
#include <stdio.h>
#include "filter.h"


#define THROW(message) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " message, __FILE__, __LINE__)

#define THROWF(fmt, ...) \
  throw_exception_("Exception thrown at %s:%d\r\nMessage: " fmt, __FILE__, __LINE__, __VA_ARGS__)

static void throw_exception_(const char* fmt, ...)
{
    char buf[300];
    va_list arg;
    va_start(arg, fmt);
    vsnprintf_s(buf, sizeof(buf), fmt, arg);
    va_end(arg);
    printf(buf);
    throw buf;
}

#define CUDA_CHECK(call) \
		do { \
			cudaError_t err__ = call; \
			if (err__ != cudaSuccess) { \
				THROWF("[CUDA Error] %d: %s", err__, cudaGetErrorString(err__)); \
			} \
		} while (0)

struct PIXEL_YCA {
    short    y;
    short    cb;
    short    cr;
    short    a; // 使用しない
};

struct Image {
    int pitch; // PIXEL_YCでのみ有効（PIXEL_YCAはパディングなし）
    int width;
    int height;
};

struct BandingParam : Image {
    int opt;
    int seed;
    int ditherY;
    int ditherC;
    int rand_each_frame;
    int sample_mode;
    int blur_first;
    int range;
    int threshold_y;
    int threshold_cb;
    int threshold_cr;
    int interlaced;
    int frame_number;
};

bool reduce_banding_cuda(BandingParam* param, PIXEL_YC* src, PIXEL_YC* dst);
