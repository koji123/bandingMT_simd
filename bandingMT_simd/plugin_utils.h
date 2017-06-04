#pragma once

#include <Windows.h>
#include <stdio.h>
#include <stdint.h>

static void init_console()
{
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	freopen("CONIN$", "r", stdin);
}

class PerformanceTimer
{
    enum {
        PRINT_CYCLE = 10,
        MAX_SECTIONS = 5
    };
public:
    PerformanceTimer()
        : times()
        , cycle()
    { }
    void start() {
        section = 0;
        QueryPerformanceCounter((LARGE_INTEGER*)&prev);
    }
    void next() {
        int64_t now;
        QueryPerformanceCounter((LARGE_INTEGER*)&now);
        times[cycle][section++] = now - prev;
        prev = now;
    }
    void end() {
        next();
        if (++cycle == PRINT_CYCLE) {
            print();
            cycle = 0;
        }
    }
private:
    int64_t times[PRINT_CYCLE][MAX_SECTIONS];
    int cycle;
    int section;
    int64_t prev;

    void print() {
        int64_t freq;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

        double total = 0.0;
        for (int i = 0; i < MAX_SECTIONS; ++i) {
            int64_t sum = 0;
            for (int c = 0; c < PRINT_CYCLE; ++c) {
                sum += times[c][i];
            }
            double avg = (double)sum / freq * 1000.0;
            total += avg;
            printf("%2d: %f ms\n", i, avg);
        }
        printf("total: %f ms\n", total);
    }
};
