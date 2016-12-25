#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

typedef vector<double> Point;
typedef vector<Point> Points;

// Gives random number in range [0..max_value]
unsigned int UniformRandom(unsigned int max_value) {
    unsigned int rnd = ((static_cast<unsigned int>(rand()) % 32768) << 17) |
                       ((static_cast<unsigned int>(rand()) % 32768) << 2) |
                       rand() % 4;
    return ((max_value + 1 == 0) ? rnd : rnd % (max_value + 1));
}

double Distance(const Point& point1, const Point& point2) {
    double distance_sqr = 0;
    size_t dimensions = point1.size();
    for (size_t i = 0; i < dimensions; ++i) {
        distance_sqr += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return distance_sqr;
}

size_t FindNearestCentroid(const Points& centroids, const Point& point) {
    double min_distance = Distance(point, centroids[0]);
    size_t centroid_index = 0;
    for (size_t i = 1; i < centroids.size(); ++i) {
        double distance = Distance(point, centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            centroid_index = i;
        }
    }

    return centroid_index;
}

// Calculates new centroid position as mean of positions of 3 random centroids
Point GetRandomPosition(const Points& centroids) {
    size_t K = centroids.size();
    int c1 = rand() % K;
    int c2 = rand() % K;
    int c3 = rand() % K;
    size_t dimensions = centroids[0].size();
    Point new_position(dimensions);
    for (size_t d = 0; d < dimensions; ++d) {
        new_position[d] = (centroids[c1][d] + centroids[c2][d] + centroids[c3][d]) / 3;
    }

    return new_position;
}

vector<size_t> KMeans(const Points& data, size_t K) {
    size_t data_size = data.size();
    size_t dimensions = data[0].size();
    vector<size_t> clusters(data_size);

    // Initialize centroids randomly at data points
    Points centroids(K);
    Points swap_centroids(K);
    for (size_t i = 0; i < K; ++i) {
        centroids[i] = data[UniformRandom(data_size - 1)];
    }
    
    size_t mismatches = data.size();
    while (mismatches > 0) {
        mismatches = 0;
        swap_centroids.assign(K, Point(dimensions)); // we have to introduce swap_centroids in order to allow one single parallel {} statemrnt instead of two distinct ones
        vector<size_t> clusters_sizes(K);

        #pragma omp parallel 
        {
            Points partial_centroids(K, Point(dimensions));
            vector<size_t> partial_clusters_sizes(K, 0);

            // since data.size() >> K and data.size() >> d, it's only sensible to parallel loops over data.size(), making other loops execute sequentially
            #pragma omp for reduction(+:mismatches)
            for (size_t i = 0; i < data_size; ++i) {
                size_t nearest_cluster = FindNearestCentroid(centroids, data[i]);
                
                if (clusters[i] != nearest_cluster) {
                    clusters[i] = nearest_cluster;
                    mismatches += 1;
                }
            
                for (size_t d = 0; d < dimensions; ++d) {
                    partial_centroids[clusters[i]][d] += data[i][d]; // private, no atomic
                }
                ++partial_clusters_sizes[clusters[i]]; // private, so no atomic
            } // here every process will wait until everyone finishes

            if (mismatches > 0) { // this check improves effectiveness from 0.90 to 0.91 :)
                // this loop has to be done by every processor to merge the results
                for (size_t i = 0; i < K; ++i) {
                    for (size_t d = 0; d < dimensions; ++d) {
                        #pragma omp atomic
                        swap_centroids[i][d] += partial_centroids[i][d];
                    }
                }
                
                for (size_t i = 0; i < K; ++i) {    
                    #pragma omp atomic
                    clusters_sizes[i] += partial_clusters_sizes[i];
                }
            }
        }

        if (mismatches == 0) {
            break;
        }

        centroids = swap_centroids;

        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] != 0) {
                for (size_t d = 0; d < dimensions; ++d) {
                    centroids[i][d] /= clusters_sizes[i];
                }
            }
        }

        for (size_t i = 0; i < K; ++i) {
            if (clusters_sizes[i] == 0) {
                centroids[i] = GetRandomPosition(centroids);
            }
        }
    }

    return clusters;
}

// void ReadPoints(Points* data, ifstream& input) {
//     size_t data_size;
//     size_t dimensions;
//     input >> data_size >> dimensions;
//     data->assign(data_size, Point(dimensions));
//     for (size_t i = 0; i < data_size; ++i) {
//         for (size_t d = 0; d < dimensions; ++d) {
//             double coord;
//             input >> coord;
//             (*data)[i][d] = coord;
//         }
//     }
// }

// optimized version using atof()
void ReadPoints(Points* data, ifstream& input) {
    size_t data_size;
    size_t dimensions;
    input >> data_size >> dimensions;
    data->assign(data_size, Point(dimensions));
    string s;
    for (size_t i = 0; i < data_size; ++i) {
        for (size_t d = 0; d < dimensions; ++d) {
            input >> s;
            (*data)[i][d] = atof(s.c_str());
        }
    }
}

void WriteOutput(const vector<size_t>& clusters, ofstream& output) {
    for (size_t i = 0; i < clusters.size(); ++i) {
        output << clusters[i] << '\n';
    }
}

int main(int argc , char** argv) {
    if (argc != 4) {
        std::printf("Usage: %s number_of_clusters input_file output_file\n", argv[0]);
        return 1;
    }
    size_t K = atoi(argv[1]);

    char* input_file = argv[2];
    ifstream input;
    input.open(input_file, ifstream::in);
    if(!input) {
        cerr << "Error: input file could not be opened\n";
        return 1;
    }

    Points data;
    ReadPoints(&data, input);
    input.close();

    char* output_file = argv[3];
    ofstream output;
    output.open(output_file, ifstream::out);
    if(!output) {
        cerr << "Error: output file could not be opened\n";
        return 1;
    }

    srand(123); // for reproducible results

    vector<size_t> clusters = KMeans(data, K);

    WriteOutput(clusters, output);
    output.close();

    return 0;
}
