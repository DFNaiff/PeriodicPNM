# Tutorial 4: Felzenszwalb-Huttenlocher Algorithm - The Math Behind EDT

## What is Euclidean Distance Transform?

**Problem**: Given a binary image, compute the distance from each background pixel to the nearest feature pixel.

**Example**:
```
Binary:          EDT:
 1 0 0 0 0       0 1 2 3 4
 0 0 0 0 0       1 1.4 2.2 3.2 4.1
 0 0 1 0 0       2 1 0 1 2
```

Where `1` = feature, `0` = background.

## Naive Approach (Slow)

```cpp
// For each background pixel:
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        float min_dist = INFINITY;

        // Check distance to ALL feature pixels
        for (int fy = 0; fy < height; ++fy) {
            for (int fx = 0; fx < width; ++fx) {
                if (binary[fy][fx] == 1) {
                    float dist = sqrt((x-fx)² + (y-fy)²);
                    min_dist = min(min_dist, dist);
                }
            }
        }

        output[y][x] = min_dist;
    }
}
```

**Complexity**: O(N² × M²) for N×M image = O(N⁴)

**Way too slow!**

## Key Insight: Separable Property

**Euclidean distance in 2D**:
```
d(x, y) = sqrt((x - fx)² + (y - fy)²)
```

**Can be computed in two passes**:
1. **Pass 1**: Compute distances along x-axis
2. **Pass 2**: Use Pass 1 results to compute final distances

**This reduces complexity from O(N⁴) to O(N²)!**

## The 1D Problem

First, solve EDT in 1D:

**Input**: Cost function `f[0..n-1]`
- `f[i] = 0` if position i is a feature
- `f[i] = ∞` otherwise

**Output**: Distance function `d[0..n-1]`
- `d[x] = min_p (|x - p| + f[p])`
- Distance from x to nearest feature

**Example**:
```
f: [0, ∞, ∞, ∞, 0, ∞, ∞]
d: [0, 1, 2, 1, 0, 1, 2]
```

## Parabola Interpretation

### Distance as a Parabola

Distance from position x to feature at position p:
```
g_p(x) = (x - p)² + f[p]
```

This is a **parabola** centered at p!

**Example**: Feature at p=2, f[2]=0
```
g_2(x) = (x - 2)²
```

Parabola:
```
x:    0   1   2   3   4   5   6
g_2:  4   1   0   1   4   9   16
```

### Multiple Features → Multiple Parabolas

If we have features at positions 0 and 4:
```
g_0(x) = (x - 0)² + 0 = x²
g_4(x) = (x - 4)² + 0
```

**The EDT is the lower envelope of all parabolas!**

```
       *         *             g_0
      * *       * *
     *   *     *   *
    *     *   *     *          g_4
   *       * *       *
  *         *         *
0   1   2   3   4   5   6
d: 0   1   2   3   4   5   6  ← minimum at each x
```

## Felzenszwalb-Huttenlocher Algorithm

### Two-Phase Algorithm

**Phase 1**: Build lower envelope
- Process features left to right
- Keep track of which parabolas are visible
- Store only the visible parabolas

**Phase 2**: Evaluate lower envelope
- For each position x, find which parabola is lowest
- Compute distance

### Data Structures

**`v[]`**: Indices of visible parabolas (up to n of them)
**`z[]`**: Intersection points between adjacent parabolas

**Example**:
```
v = [0, 4, 7]     # Parabolas centered at 0, 4, 7 are visible
z = [-∞, 2, 5.5, +∞]  # Parabola v[0] is lowest for x < 2
                       # Parabola v[1] is lowest for 2 ≤ x < 5.5
                       # Parabola v[2] is lowest for x ≥ 5.5
```

### Phase 1: Building the Envelope

**Algorithm**:
```cpp
k = 0;  // Number of parabolas in envelope
v[0] = 0;  // First parabola
z[0] = -INF;
z[1] = +INF;

for (int q = 1; q < n; ++q) {  // For each potential feature
    // Compute intersection between parabola q and last envelope parabola v[k]
    s = intersection(q, v[k]);

    // If intersection is to the left of current envelope boundary,
    // the last parabola is completely dominated - remove it
    while (k > 0 && s <= z[k]) {
        --k;
        s = intersection(q, v[k]);
    }

    // Add new parabola to envelope
    ++k;
    v[k] = q;
    z[k] = s;
    z[k+1] = +INF;
}
```

**Key idea**: As we go right, if a new parabola dominates the previous one everywhere, we remove the old one!

### Intersection Point Formula

Two parabolas:
```
g_i(x) = (x - i)² + f[i]
g_j(x) = (x - j)² + f[j]
```

Set equal:
```
(x - i)² + f[i] = (x - j)² + f[j]
x² - 2ix + i² + f[i] = x² - 2jx + j² + f[j]
```

Solve for x:
```
x = (f[i] + i² - f[j] - j²) / (2(i - j))
```

**Our code**:
```cpp
auto S = [&](int i, int j) -> float32 {
    return ((f[i] + i*1.0f*i) - (f[j] + j*1.0f*j)) / (2.0f * (i - j));
};
```

### Phase 2: Evaluating the Envelope

```cpp
k = 0;  // Start with first parabola
for (int x = 0; x < n; ++x) {
    // Find which parabola is lowest at x
    while (z[k+1] < x) {
        ++k;  // Move to next parabola
    }

    // Parabola v[k] is lowest at x
    int p = v[k];
    float dx = x - p;
    d[x] = dx*dx + f[p];
}
```

**Visualization**:
```
z = [-∞, 2, 5.5, +∞]
v = [0, 4, 7]

x=0: z[1]=2 not < 0, use v[0]=0
x=1: z[1]=2 not < 1, use v[0]=0
x=2: z[1]=2 not < 2, use v[0]=0
x=3: z[1]=2 < 3, k++, z[2]=5.5 not < 3, use v[1]=4
x=4: z[2]=5.5 not < 4, use v[1]=4
x=5: z[2]=5.5 not < 5, use v[1]=4
x=6: z[2]=5.5 < 6, k++, use v[2]=7
```

## Complete Example

**Input**:
```
f = [0, ∞, ∞, 0, ∞, ∞, ∞]
     0  1  2  3  4  5  6
```

**Features at positions 0 and 3.**

### Phase 1: Build Envelope

**Initialization**:
```
k = 0
v[0] = 0
z = [-∞, +∞]
```

**q = 1** (non-feature, f[1] = ∞):
- Intersection with v[0]=0: s = (∞ - 0) / 2 = +∞
- s > z[0], so add to envelope
```
k = 1
v = [0, 1]
z = [-∞, +∞, +∞]
```

**q = 2** (non-feature):
- Similar, add to envelope

**q = 3** (feature, f[3] = 0):
- Intersection with v[k]=2: s = (0 + 9 - ∞ - 4) / 2 = -∞ (dominated!)
- Remove v[2]
- Intersection with v[1]=1: still dominated, remove
- Intersection with v[0]=0: s = (0 + 9 - 0 - 0) / 6 = 1.5
- Add to envelope
```
k = 1
v = [0, 3]
z = [-∞, 1.5, +∞]
```

**q = 4, 5, 6** (all non-features):
- Add to envelope (but will be dominated eventually)

**Final envelope**:
```
v = [0, 3]
z = [-∞, 1.5, +∞]
```

### Phase 2: Evaluate

```
x=0: use v[0]=0, d[0] = (0-0)² = 0
x=1: use v[0]=0, d[1] = (1-0)² = 1
x=2: z[1]=1.5 < 2, use v[1]=3, d[2] = (2-3)² = 1
x=3: use v[1]=3, d[3] = (3-3)² = 0
x=4: use v[1]=3, d[4] = (4-3)² = 1
x=5: use v[1]=3, d[5] = (5-3)² = 4
x=6: use v[1]=3, d[6] = (6-3)² = 9
```

**Output**:
```
d = [0, 1, 1, 0, 1, 2, 3]
```

Perfect!

## Complexity Analysis

### Time Complexity

**Phase 1**:
- Loop over n positions: O(n)
- Each position added at most once: O(n)
- Each position removed at most once: O(n)
- **Total: O(n)**

**Phase 2**:
- Loop over n positions: O(n)
- k increments at most n times total: O(n)
- **Total: O(n)**

**Overall: O(n)** - linear time!

### Space Complexity

- `v[]`: O(n)
- `z[]`: O(n)
- **Total: O(n)**

## Our Implementation

### Non-Periodic Version

```cpp
static inline void
edt_1d_nonperiodic(const float32* f,
                   int n,
                   int* v,
                   float32* z,
                   float32* d)
{
    int k = 0;
    v[0] = 0;
    z[0] = -EDT_INF;
    z[1] =  EDT_INF;

    // Lambda for intersection
    auto S = [&](int i, int j) -> float32 {
        return ((f[i] + i*1.0f*i) - (f[j] + j*1.0f*j)) / (2.0f * (i - j));
    };

    // Phase 1: Build envelope
    for (int q = 1; q < n; ++q) {
        float32 s = S(q, v[k]);
        while (k > 0 && s <= z[k]) {
            --k;
            s = S(q, v[k]);
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k+1] = EDT_INF;
    }

    // Phase 2: Evaluate
    k = 0;
    for (int x = 0; x < n; ++x) {
        while (z[k+1] < x) {
            ++k;
        }
        int p = v[k];
        float32 dx = static_cast<float32>(x - p);
        d[x] = dx*dx + f[p];
    }
}
```

**Key points**:
1. **Lambda function** `S`: Computes intersection elegantly
2. **While loop**: Removes dominated parabolas
3. **Two-phase structure**: Clear separation

## Extension to 2D and 3D

### Separable Property

2D EDT:
```
EDT_2D(image) = EDT_1D_y(EDT_1D_x(image))
```

1. Apply 1D EDT to each column (x-direction)
2. Apply 1D EDT to result along rows (y-direction)

3D EDT:
```
EDT_3D(volume) = EDT_1D_z(EDT_1D_y(EDT_1D_x(volume)))
```

Three passes, one per dimension!

### Why This Works

**Squared Euclidean distance**:
```
d²(x, y, z) = (x - fx)² + (y - fy)² + (z - fz)²
```

**Separable**:
- First pass: partial distance from x-differences
- Second pass: adds y-differences to partial distances
- Third pass: adds z-differences to get final distances

## Key Takeaways

1. **FH algorithm is O(n)** - fastest possible for exact EDT
2. **Lower envelope of parabolas** - beautiful geometric interpretation
3. **Two phases**: build envelope, then evaluate
4. **Separable property** extends to higher dimensions
5. **Our code uses this algorithm** for all 1D EDT computations

## Next Tutorial

In **Tutorial 5**, we'll see how we modify the FH algorithm to handle **periodic boundary conditions** using a virtual domain approach!
