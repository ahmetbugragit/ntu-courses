

#include <iostream>
#include <random> // For random number generation
#include <chrono> // For timing
using namespace std;

static long long key_comparisons = 0; // Global key comparison counter
static int a[10000000];                 // static array on global/static memory

// original mergesort: merge on a[]
void merge(int n, int m) {
    if (m - n <= 0) return; // base case: 0 or 1 element

    int mid  = (n + m) / 2; // divide the list into two halves
    int size = m - n + 1;
    int* tmp = new int[size]; // temporary array for the merged list

    int i = n, j = mid + 1, k = 0; // pointer for the first half, second half & temporary array 
    
    while (i <= mid && j <= m) { // while both halves are not empty
        key_comparisons++; // count comparison
        if (a[i] <= a[j]) tmp[k++] = a[i++];
        else              tmp[k++] = a[j++];
    }
    while (i <= mid) tmp[k++] = a[i++];
    while (j <= m)   tmp[k++] = a[j++];
    for (int p = 0; p < size; ++p) a[n + p] = tmp[p];
    delete[] tmp; // free heap memory
}

void mergesort(int n, int m){
    if (m - n <= 0) return; // Base case
    int mid = (n + m) / 2;

    if (m - n > 1) { // Recursive sort and merge
        mergesort(n, mid);
        mergesort(mid + 1, m);
    }
    merge(n, m); // Merge sorted halves
}

// insertion sort (ALIST)
using KeyType = int;
using DataType = int;

class ALIST {
public:
    KeyType  key; // Key used for sorting
    DataType data; // Associated data
};

void swap(ALIST &x, ALIST &y){ ALIST t=x; x=y; y=t; }

void InsertionSort(ALIST slot[], int n){ // Simple Insertion Sort used for small subarrays
    for (int i = 1; i < n; ++i)
        for (int j = i; j > 0; --j){
            key_comparisons++; // Count comparison
            if (slot[j].key < slot[j-1].key) swap(slot[j], slot[j-1]);
            else break;
        }
}

// hybrid mergesort (ALIST)
void hybrid_mergesort(ALIST slot[], int n, int m, int S){
    if (m - n <= 0) return;
    if (m - n + 1 <= S){ InsertionSort(slot + n, m - n + 1); return; }  // Use insertion sort if size is small (≤ S)

    int mid = (n + m) / 2;
    hybrid_mergesort(slot, n, mid, S); // Recursive hybrid merge sort
    hybrid_mergesort(slot, mid + 1, m, S);

    int size = m - n + 1;
    ALIST tmp[size]; // Temporary buffer on stack

    int i = n, j = mid + 1, k = 0; // Merge step
    while (i <= mid && j <= m) {
        key_comparisons++;
        if (slot[i].key <= slot[j].key) tmp[k++] = slot[i++];
        else                             tmp[k++] = slot[j++];
    }
    while (i <= mid) tmp[k++] = slot[i++];
    while (j <= m)   tmp[k++] = slot[j++];
    for (int p = 0; p < k; ++p) slot[n + p] = tmp[p]; // Copy back to original
}

// dataset (ALIST)
void generate_dataset(ALIST arr[], int n, int xmax){
    random_device rd; // Seed for RNG
    mt19937 gen(rd()); // Mersenne Twister RNG
    uniform_int_distribution<> dist(1, xmax); // Uniform range [1, xmax]
    for (int i = 0; i < n; ++i){
        arr[i].key  = dist(gen); // Random key
        arr[i].data = i; // Just use index as data
    }
}

int main(){
    ios::sync_with_stdio(false); // Speed up cin/cout
    cin.tie(nullptr); // Untie cin from cout

    // fixed S, varying n
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};  // Input sizes to test
    int S = 32; // Threshold for switching to insertion sort
    int num_sizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int idx = 0; idx < num_sizes; ++idx){
        int n = sizes[idx];

        ALIST arr[n]; // Generate dataset of ALISTs
        generate_dataset(arr, n, n);

        for (int i = 0; i < n; ++i) a[i] = arr[i].key; // Copy keys to global int array `a` for use by mergesort

        // Display Insertion Sort
        if (n <= 10000){
            key_comparisons = 0;
            ALIST temp1[n];
            for (int i = 0; i < n; ++i) temp1[i] = arr[i];
            auto t1 = chrono::high_resolution_clock::now();
            InsertionSort(temp1, n);
            auto t2 = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double, milli>(t2 - t1).count();
            cout << "InsertionSort n=" << n
                 << ", comps=" << key_comparisons
                 << ", time(ms)=" << ms << '\n';
        } else {
            cout << "InsertionSort n=" << n << " -> too long\n";
        }

        // Display Merge Sort
        key_comparisons = 0;
        auto t3 = chrono::high_resolution_clock::now();
        mergesort(0, n - 1);
        auto t4 = chrono::high_resolution_clock::now();
        double msM = chrono::duration<double, milli>(t4 - t3).count();
        cout << "MergeSort n=" << n
             << ", comps=" << key_comparisons
             << ", time(ms)=" << msM << '\n';

        // Display Hybrid Sort
        key_comparisons = 0;
        ALIST temp2[n];
        for (int i = 0; i < n; ++i) temp2[i] = arr[i];
        auto t5 = chrono::high_resolution_clock::now();
        hybrid_mergesort(temp2, 0, n - 1, S);
        auto t6 = chrono::high_resolution_clock::now();
        double msH = chrono::duration<double, milli>(t6 - t5).count();
        cout << "HybridMergeSort n=" << n
             << ", S=" << S
             << ", comps=" << key_comparisons
             << ", time(ms)=" << msH << '\n';

        cout << "----------------------------------------\n";
    }

    // fixed n, varying S
    int n_fixed = 200000;
    int S_values[] = {4, 8, 16, 24, 32, 40, 48, 64, 96, 128};
    int num_S = sizeof(S_values)/sizeof(S_values[0]);

    ALIST base[n_fixed];
    generate_dataset(base, n_fixed, n_fixed);

    // Run MergeSort as reference
    for (int i = 0; i < n_fixed; ++i) a[i] = base[i].key;
    key_comparisons = 0;
    auto r1 = chrono::high_resolution_clock::now();
    mergesort(0, n_fixed - 1);
    auto r2 = chrono::high_resolution_clock::now();
    double msM_ref = chrono::duration<double, milli>(r2 - r1).count();
    long long compsM_ref = key_comparisons;
    cout << "MergeSort n=" << n_fixed
         << ", comps=" << compsM_ref
         << ", time(ms)=" << msM_ref << '\n';

    // Test HybridMergeSort with different S values
    for (int si = 0; si < num_S; ++si){
        int S_try = S_values[si];
        key_comparisons = 0;
        ALIST cur[n_fixed];
        for (int i = 0; i < n_fixed; ++i) cur[i] = base[i];

        auto h1 = chrono::high_resolution_clock::now();
        hybrid_mergesort(cur, 0, n_fixed - 1, S_try);
        auto h2 = chrono::high_resolution_clock::now();
        double msH_try = chrono::duration<double, milli>(h2 - h1).count();

        cout << "HybridMergeSort n=" << n_fixed
             << ", S=" << S_try
             << ", comps=" << key_comparisons
             << ", time(ms)=" << msH_try << '\n';
    }

    return 0;
}


