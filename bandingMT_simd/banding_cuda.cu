
#include "banding_cuda.h"
#include "plugin_utils.h"
#include "xor_rand.h"

#include <stdio.h>
#include <cuda_runtime.h>


static int nblocks(int n, int width) {
    return (n + width - 1) / width;
}

#ifdef ENABLE_PERF
static PerformanceTimer* g_timer = NULL;
#define TIMER_START g_timer->start()
#define TIMER_NEXT CUDA_CHECK(cudaDeviceSynchronize()); g_timer->next()
#define TIMER_END CUDA_CHECK(cudaDeviceSynchronize()); g_timer->end()
#else
#define TIMER_START
#define TIMER_NEXT
#define TIMER_END
#endif


class RandomSource
{
public:
    RandomSource(
        int width, int height, int seed, bool rand_each_frame,
        int max_per_pixel, int max_frames, int frame_skip_len)
        : width(width)
        , height(height)
        , seed(seed)
        , rand_each_frame(rand_each_frame)
        , max_per_pixel(max_per_pixel)
        , max_frames(max_frames)
        , frame_skip_len(frame_skip_len)
    {
        int length = width * (height * max_per_pixel + max_frames * frame_skip_len);
        CUDA_CHECK(cudaMalloc((void**)&dev_rand, length));

        uint8_t* rand_buf = (uint8_t*)malloc(length);

        xor128_t xor;
        xor128_init(&xor, seed);

        int i = 0;
        for (; i <= length - 4; i += 4) {
            xor128(&xor);
            *(uint32_t*)(rand_buf + i) = xor.w;
        }
        if (i < length) {
            xor128(&xor);
            memcpy(&rand_buf[i], &xor.w, length - i);
        }

        CUDA_CHECK(cudaMemcpy(dev_rand, rand_buf, length, cudaMemcpyHostToDevice));

        free(rand_buf);
    }

    ~RandomSource()
    {
        CUDA_CHECK(cudaFree(dev_rand));
    }

    const uint8_t* getRand(int frame) const
    {
        if (rand_each_frame) {
            return dev_rand + width * frame_skip_len * (frame % max_frames);
        }
        return dev_rand;
    }

    bool isSame(int owidth, int oheight, int oseed, bool orand_each_frame)
    {
        if (owidth != width ||
            oheight != height ||
            oseed != seed ||
            orand_each_frame != rand_each_frame) {
            return false;
        }
        return true;
    }

private:
    uint8_t *dev_rand;
    int width, height, seed;
    int max_per_pixel;
    int max_frames;
    int frame_skip_len;
    bool rand_each_frame;
};

__global__ void kl_convert_yc_to_yca(PIXEL_YCA* yca, PIXEL_YC* yc, int pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        PIXEL_YC s = yc[y * pitch + x];
        PIXEL_YCA d = { s.y, s.cb, s.cr };
        yca[y * width + x] = d;
    }
}

void convert_yc_to_yca(PIXEL_YCA* yca, PIXEL_YC* yc, int pitch, int width, int height)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_convert_yc_to_yca << <blocks, threads >> >(yca, yc, pitch, width, height);
}

__global__ void kl_convert_yca_to_yc(PIXEL_YC* yc, PIXEL_YCA* yca, int pitch, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        PIXEL_YCA s = yca[y * width + x];
        PIXEL_YC d = { s.y, s.cb, s.cr };
        yc[y * pitch + x] = d;
    }
}

void convert_yca_to_yc(PIXEL_YC* yc, PIXEL_YCA* yca, int pitch, int width, int height)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(width, threads.x), nblocks(height, threads.y));
    kl_convert_yca_to_yc << <blocks, threads >> >(yc, yca, pitch, width, height);
}

class PixelYC {
public:
    PixelYC(Image info, PIXEL_YC* src, PIXEL_YC* dst)
        : info(info), src(src), dst(dst)
    {
        int yc_size = info.pitch * info.height;

        CUDA_CHECK(cudaMalloc(&dev_src, yc_size * sizeof(PIXEL_YC)));
        CUDA_CHECK(cudaMalloc(&dev_dst, yc_size * sizeof(PIXEL_YC)));

        CUDA_CHECK(cudaMemcpyAsync(
            dev_src, src, yc_size * sizeof(PIXEL_YC), cudaMemcpyHostToDevice));
    }
    ~PixelYC() {
        int yc_size = info.pitch * info.height;

        CUDA_CHECK(cudaMemcpy(
            dst, dev_dst, yc_size * sizeof(PIXEL_YC), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dev_src));
        CUDA_CHECK(cudaFree(dev_dst));
    }
    PIXEL_YC* getsrc() { return dev_src; }
    PIXEL_YC* getdst() { return dev_dst; }
private:
    Image info;
    PIXEL_YC *src, *dst, *dev_src, *dev_dst;
};

class PixelYCA {
public:
    PixelYCA(Image info, PIXEL_YC* src, PIXEL_YC* dst)
        : info(info), src(src), dst(dst)
    {
        int yc_size = info.pitch * info.height;
        int yca_size = info.width * info.height;

        CUDA_CHECK(cudaMalloc(&dev_src, yc_size * sizeof(PIXEL_YC)));
        CUDA_CHECK(cudaMalloc(&dev_dsrc, yca_size * sizeof(PIXEL_YCA)));
        CUDA_CHECK(cudaMalloc(&dev_ddst, yca_size * sizeof(PIXEL_YCA)));
        CUDA_CHECK(cudaMalloc(&dev_dst, yc_size * sizeof(PIXEL_YC)));

        CUDA_CHECK(cudaMemcpyAsync(
            dev_src, src, yc_size * sizeof(PIXEL_YC), cudaMemcpyHostToDevice));

        convert_yc_to_yca(dev_dsrc, dev_src, info.pitch, info.width, info.height);
    }
    ~PixelYCA() {
        int yc_size = info.pitch * info.height;

        convert_yca_to_yc(dev_dst, dev_ddst, info.pitch, info.width, info.height);

        CUDA_CHECK(cudaMemcpy(
            dst, dev_dst, yc_size * sizeof(PIXEL_YC), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(dev_src));
        CUDA_CHECK(cudaFree(dev_dsrc));
        CUDA_CHECK(cudaFree(dev_ddst));
        CUDA_CHECK(cudaFree(dev_dst));
    }
    PIXEL_YCA* getsrc() { return (PIXEL_YCA*)dev_dsrc; }
    PIXEL_YCA* getdst() { return (PIXEL_YCA*)dev_ddst; }
private:
    Image info;
    PIXEL_YC *src, *dst, *dev_src, *dev_dst;
    PIXEL_YCA *dev_dsrc, *dev_ddst;
};

// ÉâÉìÉ_ÉÄÇ»128bitóÒÇÉâÉìÉ_ÉÄÇ» -range Å` range Ç…ÇµÇƒï‘Ç∑
// range ÇÕ0Å`127à»â∫
static __device__ char random_range(uint8_t random, char range) {
    return ((((range << 1) + 1) * (int)random) >> 8) - range;
}

static __device__ short3 get_abs_diff(short3 a, short3 b) {
    short3 diff;
    diff.x = abs(a.x - b.x);
    diff.y = abs(a.y - b.y);
    diff.z = abs(a.z - b.z);
    return diff;
}

static __device__ short4 get_abs_diff(short4 a, short4 b) {
    short4 diff;
    diff.x = abs(a.x - b.x);
    diff.y = abs(a.y - b.y);
    diff.z = abs(a.z - b.z);
    return diff;
}

static __device__ PIXEL_YCA get_abs_diff(PIXEL_YCA a, PIXEL_YCA b) {
    PIXEL_YCA diff;
    diff.y = abs(a.y - b.y);
    diff.cb = abs(a.cb - b.cb);
    diff.cr = abs(a.cr - b.cr);
    return diff;
}

static __device__ PIXEL_YC get_abs_diff(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC diff;
    diff.y = abs(a.y - b.y);
    diff.cb = abs(a.cb - b.cb);
    diff.cr = abs(a.cr - b.cr);
    return diff;
}

template <typename T>
static __device__ T get_min(T a, T b) {
    return a < b ? a : b;
}

template <typename T>
static __device__ T get_max(T a, T b) {
    return a > b ? a : b;
}

template <>
static __device__ short3 get_max(short3 a, short3 b) {
    short3 max_value;
    max_value.x = get_max(a.x, b.x);
    max_value.y = get_max(a.y, b.y);
    max_value.z = get_max(a.z, b.z);
    return max_value;
}

template <>
static __device__ short4 get_max(short4 a, short4 b) {
    short4 max_value;
    max_value.x = get_max(a.x, b.x);
    max_value.y = get_max(a.y, b.y);
    max_value.z = get_max(a.z, b.z);
    return max_value;
}

template <>
static __device__ PIXEL_YCA get_max(PIXEL_YCA a, PIXEL_YCA b) {
    PIXEL_YCA max_value;
    max_value.y = get_max(a.y, b.y);
    max_value.cb = get_max(a.cb, b.cb);
    max_value.cr = get_max(a.cr, b.cr);
    return max_value;
}

template <>
static __device__ PIXEL_YC get_max(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC max_value;
    max_value.y = get_max(a.y, b.y);
    max_value.cb = get_max(a.cb, b.cb);
    max_value.cr = get_max(a.cr, b.cr);
    return max_value;
}

template <typename T>
static __device__ T get_min(T a, T b, T c, T d) {
    return get_min(get_min(a, b), get_min(c, d));
}

template <typename T>
static __device__ T get_max(T a, T b, T c, T d) {
    return get_max(get_max(a, b), get_max(c, d));
}

static __device__ short3 get_avg(short3 a, short3 b) {
    short3 avg;
    avg.x = (a.x + b.x + 1) >> 1;
    avg.y = (a.y + b.y + 1) >> 1;
    avg.z = (a.z + b.z + 1) >> 1;
    return avg;
}

static __device__ short4 get_avg(short4 a, short4 b) {
    short4 avg;
    avg.x = (a.x + b.x + 1) >> 1;
    avg.y = (a.y + b.y + 1) >> 1;
    avg.z = (a.z + b.z + 1) >> 1;
    return avg;
}

static __device__ PIXEL_YC get_avg(PIXEL_YC a, PIXEL_YC b) {
    PIXEL_YC avg;
    avg.y = (a.y + b.y + 1) >> 1;
    avg.cb = (a.cb + b.cb + 1) >> 1;
    avg.cr = (a.cr + b.cr + 1) >> 1;
    return avg;
}

static __device__ PIXEL_YCA get_avg(PIXEL_YCA a, PIXEL_YCA b) {
    PIXEL_YCA avg;
    avg.y = (a.y + b.y + 1) >> 1;
    avg.cb = (a.cb + b.cb + 1) >> 1;
    avg.cr = (a.cr + b.cr + 1) >> 1;
    return avg;
}

static __device__ short3 get_avg(short3 a, short3 b, short3 c, short3 d) {
    short3 avg;
    avg.x = (a.x + b.x + c.x + d.x + 2) >> 2;
    avg.y = (a.y + b.y + c.y + d.y + 2) >> 2;
    avg.z = (a.z + b.z + c.z + d.z + 2) >> 2;
    return avg;
}

static __device__ short4 get_avg(short4 a, short4 b, short4 c, short4 d) {
    short4 avg;
    avg.x = (a.x + b.x + c.x + d.x + 2) >> 2;
    avg.y = (a.y + b.y + c.y + d.y + 2) >> 2;
    avg.z = (a.z + b.z + c.z + d.z + 2) >> 2;
    return avg;
}

static __device__ PIXEL_YC get_avg(PIXEL_YC a, PIXEL_YC b, PIXEL_YC c, PIXEL_YC d) {
    PIXEL_YC avg;
    avg.y = (a.y + b.y + c.y + d.y + 2) >> 2;
    avg.cb = (a.cb + b.cb + c.cb + d.cb + 2) >> 2;
    avg.cr = (a.cr + b.cr + c.cr + d.cr + 2) >> 2;
    return avg;
}

static __device__ PIXEL_YCA get_avg(PIXEL_YCA a, PIXEL_YCA b, PIXEL_YCA c, PIXEL_YCA d) {
    PIXEL_YCA avg;
    avg.y = (a.y + b.y + c.y + d.y + 2) >> 2;
    avg.cb = (a.cb + b.cb + c.cb + d.cb + 2) >> 2;
    avg.cr = (a.cr + b.cr + c.cr + d.cr + 2) >> 2;
    return avg;
}

template <int sample_mode, bool blur_first>
__global__ void kl_reduce_banding(BandingParam prm, short4* __restrict__ dst, const short4* __restrict__ src, const uint8_t* __restrict__ rand)
{
    const int ditherY = prm.ditherY;
    const int ditherC = prm.ditherC;
    const int width = prm.width;
    const int height = prm.height;
    const int range = prm.range;
    const int threshold_y = prm.threshold_y;
    const int threshold_cb = prm.threshold_cb;
    const int threshold_cr = prm.threshold_cr;
    const int field_mask = prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * width + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        short4 src_val = src[offset];
        short4 avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * width + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * width + refB;

            short4 ref_p = src[offset + ref];
            short4 ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * width + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * width;

            short4 ref_0p = src[offset + ref_0];
            short4 ref_0m = src[offset - ref_0];
            short4 ref_1p = src[offset + ref_1];
            short4 ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        short4 dst_val;
        dst_val.x = (diff.x < threshold_y) ? avg.x : src_val.x;
        dst_val.y = (diff.y < threshold_cb) ? avg.y : src_val.y;
        dst_val.z = (diff.z < threshold_cr) ? avg.z : src_val.z;

        dst_val.x += random_range(rand[offset + rand_step * 2], ditherY);
        dst_val.y += random_range(rand[offset + rand_step * 3], ditherC);
        dst_val.z += random_range(rand[offset + rand_step * 4], ditherC);

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first>
void run_reduce_banding(BandingParam * prm, short4* dev_dst, const short4* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding<sample_mode, blur_first>
        << <blocks, threads, 0, stream >> >(*prm, dev_dst, dev_src, dev_rand);
}

template <int sample_mode, bool blur_first>
__global__ void kl_reduce_banding_short3(BandingParam prm, short3* __restrict__ dst, const short3* __restrict__ src, const uint8_t* __restrict__ rand)
{
    const int ditherY = prm.ditherY;
    const int ditherC = prm.ditherC;
    const int pitch = prm.pitch;
    const int width = prm.width;
    const int height = prm.height;
    const int range = prm.range;
    const int threshold_y = prm.threshold_y;
    const int threshold_cb = prm.threshold_cb;
    const int threshold_cr = prm.threshold_cr;
    const int field_mask = prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * pitch + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        short3 src_val = src[offset];
        short3 avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            short3 ref_p = src[offset + ref];
            short3 ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * pitch + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * pitch;

            short3 ref_0p = src[offset + ref_0];
            short3 ref_0m = src[offset - ref_0];
            short3 ref_1p = src[offset + ref_1];
            short3 ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        short3 dst_val;
        dst_val.x = (diff.x < threshold_y) ? avg.x : src_val.x;
        dst_val.y = (diff.y < threshold_cb) ? avg.y : src_val.y;
        dst_val.z = (diff.z < threshold_cr) ? avg.z : src_val.z;

        dst_val.x += random_range(rand[offset + rand_step * 2], ditherY);
        dst_val.y += random_range(rand[offset + rand_step * 3], ditherC);
        dst_val.z += random_range(rand[offset + rand_step * 4], ditherC);

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first>
void run_reduce_banding_short3(BandingParam * prm, short3* dev_dst, const short3* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding_short3<sample_mode, blur_first>
        << <blocks, threads, 0, stream >> >(*prm, dev_dst, dev_src, dev_rand);
}

template <int sample_mode, bool blur_first>
__global__ void kl_reduce_banding_YCA(BandingParam prm, PIXEL_YCA* __restrict__ dst, const PIXEL_YCA* __restrict__ src, const uint8_t* __restrict__ rand)
{
    const int ditherY = prm.ditherY;
    const int ditherC = prm.ditherC;
    const int width = prm.width;
    const int height = prm.height;
    const int range = prm.range;
    const int threshold_y = prm.threshold_y;
    const int threshold_cb = prm.threshold_cb;
    const int threshold_cr = prm.threshold_cr;
    const int field_mask = prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * width + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        PIXEL_YCA src_val = src[offset];
        PIXEL_YCA avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * width + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * width + refB;

            PIXEL_YCA ref_p = src[offset + ref];
            PIXEL_YCA ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * width + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * width;

            PIXEL_YCA ref_0p = src[offset + ref_0];
            PIXEL_YCA ref_0m = src[offset - ref_0];
            PIXEL_YCA ref_1p = src[offset + ref_1];
            PIXEL_YCA ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        PIXEL_YCA dst_val;
        dst_val.y = (diff.y < threshold_y) ? avg.y : src_val.y;
        dst_val.cb = (diff.cb < threshold_cb) ? avg.cb : src_val.cb;
        dst_val.cr = (diff.cr < threshold_cr) ? avg.cr : src_val.cr;

        dst_val.y += random_range(rand[offset + rand_step * 2], ditherY);
        dst_val.cb += random_range(rand[offset + rand_step * 3], ditherC);
        dst_val.cr += random_range(rand[offset + rand_step * 4], ditherC);

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first>
void run_reduce_banding_YCA(BandingParam * prm, PIXEL_YCA* dev_dst, const PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding_YCA<sample_mode, blur_first>
        << <blocks, threads, 0, stream >> >(*prm, dev_dst, dev_src, dev_rand);
}

template <int sample_mode, bool blur_first>
__global__ void kl_reduce_banding_YC(BandingParam prm, PIXEL_YC* __restrict__ dst, const PIXEL_YC* __restrict__ src, const uint8_t* __restrict__ rand)
{
    const int ditherY = prm.ditherY;
    const int ditherC = prm.ditherC;
    const int pitch = prm.pitch;
    const int width = prm.width;
    const int height = prm.height;
    const int range = prm.range;
    const int threshold_y = prm.threshold_y;
    const int threshold_cb = prm.threshold_cb;
    const int threshold_cr = prm.threshold_cr;
    const int field_mask = prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * pitch + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        PIXEL_YC src_val = src[offset];
        PIXEL_YC avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            PIXEL_YC ref_p = src[offset + ref];
            PIXEL_YC ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * pitch + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * pitch;

            PIXEL_YC ref_0p = src[offset + ref_0];
            PIXEL_YC ref_0m = src[offset - ref_0];
            PIXEL_YC ref_1p = src[offset + ref_1];
            PIXEL_YC ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        PIXEL_YC dst_val;
        dst_val.y = (diff.y < threshold_y) ? avg.y : src_val.y;
        dst_val.cb = (diff.cb < threshold_cb) ? avg.cb : src_val.cb;
        dst_val.cr = (diff.cr < threshold_cr) ? avg.cr : src_val.cr;

        dst_val.y += random_range(rand[offset + rand_step * 2], ditherY);
        dst_val.cb += random_range(rand[offset + rand_step * 3], ditherC);
        dst_val.cr += random_range(rand[offset + rand_step * 4], ditherC);

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first>
void run_reduce_banding_YC(BandingParam * prm, PIXEL_YC* dev_dst, const PIXEL_YC* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding_YC<sample_mode, blur_first>
        << <blocks, threads, 0, stream >> >(*prm, dev_dst, dev_src, dev_rand);
}

template <int sample_mode, bool blur_first>
__global__ void kl_reduce_banding_naive(BandingParam prm, PIXEL_YC* dst, const PIXEL_YC* src, const uint8_t* rand)
{
    const int ditherY = prm.ditherY;
    const int ditherC = prm.ditherC;
    const int pitch = prm.pitch;
    const int width = prm.width;
    const int height = prm.height;
    const int range = prm.range;
    const int threshold_y = prm.threshold_y;
    const int threshold_cb = prm.threshold_cb;
    const int threshold_cr = prm.threshold_cr;
    const int field_mask = prm.interlaced ? 0xfe : 0xff;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int rand_step = width * height;
    const int offset = y * pitch + x;

    if (x < width && y < height) {

        const int range_limited = get_min(range,
            get_min(y, height - y - 1, x, width - x - 1));
        const char refA = random_range(rand[offset + rand_step * 0], range_limited);
        const char refB = random_range(rand[offset + rand_step * 1], range_limited);

        PIXEL_YC src_val = src[offset];
        PIXEL_YC avg, diff;

        if (sample_mode == 0) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            avg = src[offset + ref];
            diff = get_abs_diff(src_val, avg);

        }
        else if (sample_mode == 1) {
            const int ref = (char)(refA & field_mask) * pitch + refB;

            PIXEL_YC ref_p = src[offset + ref];
            PIXEL_YC ref_m = src[offset - ref];

            avg = get_avg(ref_p, ref_m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_p),
                    get_abs_diff(src_val, ref_m));
        }
        else {
            const int ref_0 = (char)(refA & field_mask) * pitch + refB;
            const int ref_1 = refA - (char)(refB & field_mask) * pitch;

            PIXEL_YC ref_0p = src[offset + ref_0];
            PIXEL_YC ref_0m = src[offset - ref_0];
            PIXEL_YC ref_1p = src[offset + ref_1];
            PIXEL_YC ref_1m = src[offset - ref_1];

            avg = get_avg(ref_0p, ref_0m, ref_1p, ref_1m);
            diff = blur_first
                ? get_abs_diff(src_val, avg)
                : get_max(get_abs_diff(src_val, ref_0p),
                    get_abs_diff(src_val, ref_0m),
                    get_abs_diff(src_val, ref_1p),
                    get_abs_diff(src_val, ref_1m));
        }

        PIXEL_YC dst_val;
        dst_val.y = (diff.y < threshold_y) ? avg.y : src_val.y;
        dst_val.cb = (diff.cb < threshold_cb) ? avg.cb : src_val.cb;
        dst_val.cr = (diff.cr < threshold_cr) ? avg.cr : src_val.cr;

        dst_val.y += random_range(rand[offset + rand_step * 2], ditherY);
        dst_val.cb += random_range(rand[offset + rand_step * 3], ditherC);
        dst_val.cr += random_range(rand[offset + rand_step * 4], ditherC);

        dst[offset] = dst_val;
    }
}

template <int sample_mode, bool blur_first>
void run_reduce_banding_naive(BandingParam * prm, PIXEL_YC* dev_dst, const PIXEL_YC* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    dim3 threads(32, 16);
    dim3 blocks(nblocks(prm->width, threads.x), nblocks(prm->height, threads.y));
    kl_reduce_banding_naive<sample_mode, blur_first>
        << <blocks, threads, 0, stream >> >(*prm, dev_dst, dev_src, dev_rand);
}

void reduce_banding(BandingParam * prm, PIXEL_YCA* dev_dst, const PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    void(*kernel_table[3][2])(BandingParam * prm, short4* dev_dst, const short4* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            run_reduce_banding<0, false>,
            NULL
        },
        {
            run_reduce_banding<1, false>,
            run_reduce_banding<1, true>,
        },
        {
            run_reduce_banding<2, false>,
            run_reduce_banding<2, true>,
        }
    };
    void(*table_YCA[3][2])(BandingParam * prm, PIXEL_YCA* dev_dst, const PIXEL_YCA* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            run_reduce_banding_YCA<0, false>,
            NULL
        },
        {
            run_reduce_banding_YCA<1, false>,
            run_reduce_banding_YCA<1, true>,
        },
        {
            run_reduce_banding_YCA<2, false>,
            run_reduce_banding_YCA<2, true>,
        }
    };

    bool blur_first = (prm->sample_mode != 0) && (prm->blur_first != 0);

    switch (prm->opt) {
    case 3:
        table_YCA[prm->sample_mode][blur_first](prm, dev_dst, dev_src, dev_rand, stream);
        break;
    case 5:
        kernel_table[prm->sample_mode][blur_first](prm, (short4*)dev_dst, (const short4*)dev_src, dev_rand, stream);
        break;
    default:
        THROW("invalid opt error");
    }
}

void reduce_banding(BandingParam * prm, PIXEL_YC* dev_dst, const PIXEL_YC* dev_src, const uint8_t* dev_rand, cudaStream_t stream)
{
    void(*table_short3[3][2])(BandingParam * prm, short3* dev_dst, const short3* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            run_reduce_banding_short3<0, false>,
            NULL
        },
        {
            run_reduce_banding_short3<1, false>,
            run_reduce_banding_short3<1, true>,
        },
        {
            run_reduce_banding_short3<2, false>,
            run_reduce_banding_short3<2, true>,
        }
    };
    void(*table_YC[3][2])(BandingParam * prm, PIXEL_YC* dev_dst, const PIXEL_YC* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            run_reduce_banding_YC<0, false>,
            NULL
        },
        {
            run_reduce_banding_YC<1, false>,
            run_reduce_banding_YC<1, true>,
        },
        {
            run_reduce_banding_YC<2, false>,
            run_reduce_banding_YC<2, true>,
        }
    };
    void(*table_naive[3][2])(BandingParam * prm, PIXEL_YC* dev_dst, const PIXEL_YC* dev_src, const uint8_t* dev_rand, cudaStream_t stream) = {
        {
            run_reduce_banding_naive<0, false>,
            NULL
        },
        {
            run_reduce_banding_naive<1, false>,
            run_reduce_banding_naive<1, true>,
        },
        {
            run_reduce_banding_naive<2, false>,
            run_reduce_banding_naive<2, true>,
        }
    };

    bool blur_first = (prm->sample_mode != 0) && (prm->blur_first != 0);

    switch (prm->opt) {
    case 1:
        table_naive[prm->sample_mode][blur_first](prm, dev_dst, dev_src, dev_rand, stream);
        break;
    case 2:
        table_YC[prm->sample_mode][blur_first](prm, dev_dst, dev_src, dev_rand, stream);
        break;
    case 4:
        table_short3[prm->sample_mode][blur_first](prm, (short3*)dev_dst, (const short3*)dev_src, dev_rand, stream);
        break;
    default:
        THROW("invalid opt error");
    }
}

class ReduceBandingInternal
{
public:
    ReduceBandingInternal(BandingParam* prm)
        : rand(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0, 5, 16, 200)
    { }
    const uint8_t* getRand(int frame) const {
        return rand.getRand(frame);
    }
    bool isSame(BandingParam* prm) {
        return rand.isSame(prm->width, prm->height, prm->seed, prm->rand_each_frame != 0);
    }
private:
    RandomSource rand;
};

static ReduceBandingInternal* data;

bool reduce_banding_cuda(BandingParam* prm, PIXEL_YC* src, PIXEL_YC* dst)
{
#ifdef ENABLE_PERF
    if (g_timer == NULL) {
        g_timer = new PerformanceTimer();
    }
#endif
    try {
        TIMER_START;
        switch (prm->opt) {
        case 1: // naive
        case 2: // PIXEL_YC
        case 4: // short3
            {
                PixelYC pixelYCA(*prm, src, dst);
                TIMER_NEXT;
                if (data == NULL) {
                    data = new ReduceBandingInternal(prm);
                }
                else if (data->isSame(prm) == false) {
                    delete data;
                    data = new ReduceBandingInternal(prm);
                }
                TIMER_NEXT;
                reduce_banding(prm, pixelYCA.getdst(),
                    pixelYCA.getsrc(), data->getRand(prm->frame_number), NULL);
                TIMER_NEXT;
            }
            break;
        case 3: // PIXEL_YCA
        case 5: // short4
            {
                PixelYCA pixelYCA(*prm, src, dst);
                TIMER_NEXT;
                if (data == NULL) {
                    data = new ReduceBandingInternal(prm);
                }
                else if (data->isSame(prm) == false) {
                    delete data;
                    data = new ReduceBandingInternal(prm);
                }
                TIMER_NEXT;
                reduce_banding(prm, pixelYCA.getdst(),
                    pixelYCA.getsrc(), data->getRand(prm->frame_number), NULL);
                TIMER_NEXT;
            }
            break;
        }
        TIMER_END;
        return true;
    }
    catch (const char*) {}
    return false;
}
