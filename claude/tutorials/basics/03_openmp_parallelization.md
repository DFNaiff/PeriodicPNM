# Tutorial 3: OpenMP Parallelization - Multi-Core Computing

## What is OpenMP?

**OpenMP (Open Multi-Processing)** is an API for parallel programming on shared-memory multi-core systems. It uses **compiler directives** (pragmas) to parallelize code with minimal changes.

### Why OpenMP?

1. **Easy to use**: Add `#pragma omp parallel for` → instant parallelization!
2. **Portable**: Works on Linux, macOS, Windows
3. **Scalable**: Automatically uses all CPU cores
4. **Low overhead**: Minimal performance penalty
5. **Incremental**: Can parallelize one loop at a time

### Your CPU

Modern CPUs have multiple cores:
- **Cores**: Independent processors on one chip
- **Threads**: Each core can run 1-2 threads simultaneously
- **Example**: 8-core CPU → 8-16 threads

**Check your system**:
```bash
lscpu | grep "^CPU(s):"  # Linux
sysctl -n hw.ncpu        # macOS
```

## OpenMP Basics

### Sequential vs Parallel Execution

**Sequential** (1 core):
```cpp
for (int i = 0; i < 1000; ++i) {
    process(i);  // Processes 0, 1, 2, ..., 999 one by one
}
// Time: T
```

**Parallel** (4 cores):
```cpp
#pragma omp parallel for
for (int i = 0; i < 1000; ++i) {
    process(i);  // Each core processes ~250 iterations
}
// Time: ~T/4
```

OpenMP divides work among cores:
- Core 0: processes 0-249
- Core 1: processes 250-499
- Core 2: processes 500-749
- Core 3: processes 750-999

### The `#pragma` Directive

```cpp
#pragma omp parallel for
```

**Breaking it down**:
- `#pragma`: Compiler directive (instruction to compiler)
- `omp`: OpenMP namespace
- `parallel for`: Parallelize the following for loop

**What the compiler does**:
1. Creates a thread team
2. Distributes loop iterations among threads
3. Each thread executes its iterations
4. Waits for all threads to finish (implicit barrier)

## Thread Management

### Thread Creation

```cpp
#pragma omp parallel
{
    // This block runs on multiple threads
    int tid = omp_get_thread_num();
    printf("Hello from thread %d\n", tid);
}
```

**Output** (4 threads):
```
Hello from thread 0
Hello from thread 2
Hello from thread 1
Hello from thread 3
```

(Order is non-deterministic!)

### Number of Threads

OpenMP automatically uses all available cores:

```cpp
#include <omp.h>

int num_threads = omp_get_max_threads();  // e.g., 8
```

**Control thread count**:
```bash
export OMP_NUM_THREADS=4  # Use 4 threads
python script.py
```

Or in code:
```cpp
omp_set_num_threads(4);  // Use 4 threads
```

## Parallel For Loops

### Basic Syntax

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    // Each iteration is independent
    array[i] = compute(i);
}
```

**Requirements**:
1. Loop iterations must be **independent**
2. Loop bounds must be known before loop starts
3. Index must be integer type

### Work Distribution

OpenMP divides iterations among threads. **Default schedule: static**

**Example**: 100 iterations, 4 threads
- Thread 0: 0-24
- Thread 1: 25-49
- Thread 2: 50-74
- Thread 3: 75-99

### Scheduling Strategies

```cpp
#pragma omp parallel for schedule(static)
#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for schedule(guided)
```

**Static** (default):
- Fixed chunk size per thread
- Low overhead
- Best for uniform work

**Dynamic**:
- Threads grab chunks on-the-fly
- Higher overhead
- Best for non-uniform work

**Guided**:
- Chunk size decreases over time
- Balance between static and dynamic

## Our Code: 2D EDT Parallelization

### Axis 0: Parallelizing Over Columns

```cpp
// Axis 0: lines along y, one line per x (columns)
#ifdef _OPENMP
#pragma omp parallel
#endif
{
    // Each thread needs its own buffers!
    std::vector<float32> f_line(ny);
    std::vector<float32> d_line(ny);
    std::vector<int> v(per0 ? 2*ny : ny);
    std::vector<float32> z(per0 ? 2*ny + 1 : ny + 1);

#ifdef _OPENMP
#pragma omp for
#endif
    for (int x = 0; x < nx; ++x) {
        // Process column x
        // ...
    }
}
```

**What's happening**:

1. **`#pragma omp parallel`**: Create thread team
   - Each thread executes the entire block
   - But they work on different iterations

2. **Thread-local buffers**:
   ```cpp
   std::vector<float32> f_line(ny);  // Each thread gets its own!
   ```
   - These are **automatic variables** (on stack)
   - Each thread has its own stack
   - No data races!

3. **`#pragma omp for`**: Distribute loop iterations
   - Thread 0 might process columns 0, 1, 2, ...
   - Thread 1 might process columns 10, 11, 12, ...
   - etc.

### Why Separate `parallel` and `for`?

**Combined** (simpler):
```cpp
#pragma omp parallel for
for (int x = 0; x < nx; ++x) {
    std::vector<float32> f_line(ny);  // Created nx times!
    // ...
}
```

**Separated** (our code, more efficient):
```cpp
#pragma omp parallel
{
    std::vector<float32> f_line(ny);  // Created once per thread!
    #pragma omp for
    for (int x = 0; x < nx; ++x) {
        // Reuse f_line for each iteration
    }
}
```

**Benefit**: Allocate buffers once per thread instead of once per iteration!

## Data Races and Thread Safety

### What is a Data Race?

Two threads access the same memory, at least one writes, without synchronization:

```cpp
int counter = 0;

#pragma omp parallel for
for (int i = 0; i < 1000; ++i) {
    counter++;  // ❌ DATA RACE!
}
// counter might be 532 instead of 1000!
```

**Why?**
```
Thread 0: read counter (0) → add 1 → write (1)
Thread 1: read counter (0) → add 1 → write (1)  ← overwrites!
// Result: counter = 1, not 2
```

### Our Code is Race-Free

**Key insight**: Each column/row is processed by only one thread!

```cpp
#pragma omp for
for (int x = 0; x < nx; ++x) {
    for (int y = 0; y < ny; ++y) {
        D[y*nx + x] = ...;  // Only this thread writes to column x
    }
}
```

**Why safe**:
- Thread 0 writes to `D[0:ny, 0]` (column 0)
- Thread 1 writes to `D[0:ny, 1]` (column 1)
- No overlap → no race!

### Thread-Local Variables

Variables declared inside `#pragma omp parallel` block are **thread-private**:

```cpp
#pragma omp parallel
{
    std::vector<float32> f_line(ny);  // Each thread has its own copy
    // ...
}
```

**Memory layout**:
```
Thread 0 stack: f_line[0..ny-1]
Thread 1 stack: f_line[0..ny-1]
Thread 2 stack: f_line[0..ny-1]
...
```

Completely separate! No sharing, no races.

## 3D EDT Parallelization

### Nested Loops

```cpp
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
        // Process line along z at fixed (y, x)
    }
}
```

**`collapse(2)`**: Parallelize over both `y` and `x` loops

**Effect**:
- Without collapse: parallelize over `y` (ny iterations)
- With collapse: parallelize over `y*x` (ny*nx iterations)

**Benefit**: More iterations → better load balancing

### Example

**Without collapse** (4 threads, ny=4, nx=100):
```
Thread 0: y=0, all x (100 iterations)
Thread 1: y=1, all x (100 iterations)
Thread 2: y=2, all x (100 iterations)
Thread 3: y=3, all x (100 iterations)
```

**With collapse**:
```
Thread 0: y=0,x=0..99, y=1,x=0..24  (125 iterations)
Thread 1: y=1,x=25..149               (125 iterations)
Thread 2: y=2,x=150..274              (125 iterations)
Thread 3: y=3,x=275..399              (125 iterations)
```

Better load balancing!

## Compilation

### Compiler Flags

**Enable OpenMP**:
```bash
# GCC/Clang
g++ -fopenmp code.cpp

# MSVC
cl /openmp code.cpp
```

**Our setup.py**:
```python
openmp_compile_args = ["-fopenmp", "-O3"]
openmp_link_args = ["-fopenmp"]
```

Both compile and link need OpenMP flag!

### Conditional Compilation

```cpp
#ifdef _OPENMP
#pragma omp parallel for
#endif
for (int i = 0; i < N; ++i) {
    // ...
}
```

**Why?**
- If OpenMP not available, code still compiles
- Just runs sequentially
- Graceful degradation

## Performance Analysis

### Speedup

**Ideal speedup** with `P` cores:
```
Speedup = T_sequential / T_parallel = P
```

**Our 128³ test**:
- Sequential estimate: ~0.5 seconds
- Parallel (8 cores): 0.066 seconds
- Speedup: ~7.5x (excellent!)

### Amdahl's Law

Not all code can be parallelized:

```
Speedup_max = 1 / (S + P/N)
```

Where:
- `S`: Serial fraction
- `P`: Parallel fraction
- `N`: Number of cores

**Our code**:
- Initialization: ~5% serial
- EDT loops: ~95% parallel
- Max speedup (8 cores): ~6.8x
- Achieved: ~7.5x (even better due to cache effects!)

### Overhead

OpenMP has overhead:
- Thread creation: ~10μs per thread
- Synchronization: ~1μs per barrier

**When parallelization doesn't help**:
```cpp
#pragma omp parallel for  // Overhead > benefit!
for (int i = 0; i < 10; ++i) {
    x[i] = i * 2;  // Too simple, too few iterations
}
```

**Rule of thumb**: Parallelize loops with >1000 iterations and non-trivial work.

## Advanced: Critical Sections and Atomics

### Critical Section

Only one thread executes at a time:

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    float local = compute(i);

    #pragma omp critical
    {
        global_sum += local;  // Only one thread at a time
    }
}
```

**Use when**: Rare updates to shared state

### Atomic Operations

Faster for simple operations:

```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    #pragma omp atomic
    counter++;  // Hardware-supported atomic increment
}
```

**Our code doesn't need these** because we have no shared writes!

## Practical Tips

### 1. **Measure Performance**

Always profile:
```cpp
auto start = std::chrono::high_resolution_clock::now();
// ... parallel code ...
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

### 2. **Check Thread Count**

```cpp
#pragma omp parallel
{
    #pragma omp single
    printf("Using %d threads\n", omp_get_num_threads());
}
```

### 3. **Avoid False Sharing**

**Bad**:
```cpp
int counters[4];  // Adjacent in memory

#pragma omp parallel for
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 1000000; ++j) {
        counters[i]++;  // Cache line bouncing!
    }
}
```

**Good**:
```cpp
int counters[4 * 16];  // 16x spacing (cache line size)

#pragma omp parallel for
for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 1000000; ++j) {
        counters[i * 16]++;  // No false sharing!
    }
}
```

## Key Takeaways

1. **OpenMP makes parallelization easy** with simple pragmas
2. **`#pragma omp parallel for`** parallelizes loops automatically
3. **Thread-local variables** prevent data races
4. **`collapse(N)`** parallelizes nested loops
5. **Overhead matters** - only parallelize large loops
6. **Our EDT code scales** nearly linearly with cores

## Next Tutorial

In **Tutorial 4**, we'll dive deep into the Felzenszwalb-Huttenlocher algorithm - the exact linear-time EDT algorithm we use!
