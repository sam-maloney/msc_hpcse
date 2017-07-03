#include <omp.h>

class omp_mutex
{
public:
    omp_mutex() { omp_init_lock(&_mutex); }
    ~omp_mutex() { omp_destroy_lock(&_mutex); }
    void lock() { omp_set_lock(&_mutex); }
    void unlock() { omp_unset_lock(&_mutex); }
private:
    omp_lock_t _mutex;
};

class omp_recursive_mutex
{
public:
    omp_recursive_mutex() { omp_init_nest_lock(&_mutex); }
    ~omp_recursive_mutex() { omp_destroy_nest_lock(&_mutex); }
    void lock() { omp_set_nest_lock(&_mutex); }
    void unlock() { omp_unset_nest_lock(&_mutex); }
private:
    omp_nest_lock_t _mutex;
};
