#include <iostream>
#include <utility>
#include <vector>
#include <set>
#include <math.h>
#include <queue>
#include <algorithm>
#include <utility>
#include <unordered_set>
#include <iterator>
#include <fstream>
#include <numeric>
#include <chrono>

class Node {
public:
    std::vector<int> data;
    std::vector<std::vector<Node *> > neighbors;
    int level;

    Node(const std::vector<int> &d, const std::vector<std::vector<Node *> > &n, int l) {
        data = d;
        neighbors = n;
        level = l;
    }
};

class HNSW {
private:
    std::vector<std::vector<int> > data;
    Node *enter_point = nullptr;

    // hyper parameters
    int m = 10;                                      // number of neighbors to connect in algo1
    int m_max = 20;                                  // limit maximum number of neighbors in algo1
    int m_max_0 = 20;                                // limit maximum number of neighbors at layer0 in algo1
    int ef_construction = 40;                        // size of dynamic candidate list
    float ml = 1.0;                                  // normalization factor for level generation
    int distance_calculation_count = 0;              // count number of calling distance function
    std::string select_neighbors_mode = "simple"; // select which select neighbor algorithm to use

    float dist_l2(const std::vector<int> *v1, const std::vector<int> *v2) {
        if (v1->size() != v2->size()) {
            throw std::runtime_error("dist_l2: vectors sizes do not match");
        }
        distance_calculation_count++;
        float dist = 0;
        for (size_t i = 0; i < v1->size(); i++) {
            dist += ((*v1)[i] - (*v2)[i]) * ((*v1)[i] - (*v2)[i]);
        }
        return sqrt(dist);
    }

public:
    int get_distance_calculation_count() const {
        return distance_calculation_count;
    }

    void set_distance_calculation_count(int set_count) {
        distance_calculation_count = set_count;
    }

    void print_graph_parameters() {
        std::cout << "m=" << m << ", m_max=" << m_max << ", m_max_0=" << m_max_0 << ", ef_construction="
                  << ef_construction << ", ml=" << ml << ", select_neighbor=" << select_neighbors_mode << std::endl;
    }

    static void log_progress(int curr, int total) {
        int barWidth = 70;
        if (curr % (total / 100) != 0) {
            return;
        }
        float progress = (float) curr / total;
        std::cout << std::flush << "\r";
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos)
                std::cout << "=";
            else if (i == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0);

        if (curr == total) {
            std::cout << std::endl;
        }
    }

    void build_graph(const std::vector<std::vector<int> > &input) {
        data = input;
        std::cout << "building graph" << std::endl;

        for (int i = 0; i < input.size(); i++) {
            //for (const std::vector<int> &i: input) {
            Node *node = new Node(input[i], std::vector<std::vector<Node *> >(), 0);

            // special case: the first node has no enter point to insert
            if (enter_point == nullptr) {
                enter_point = node;
                node->neighbors.resize(1);
                continue;
            }

            insert(node, m, m_max, m_max_0, ef_construction, ml);

            log_progress(i + 1, input.size());
        }
    }

    void insert(Node *q, int m, int m_max, int m_max_0, int ef_construction, float ml) {
        std::priority_queue<std::pair<float, Node *> > w;
        Node *ep = this->enter_point;
        int l = ep->level;
        int l_new = floor(-log((float) rand() / (RAND_MAX + 1.0)) * ml);

        // update fields of node
        q->level = l_new;
        while (q->neighbors.size() <= l_new) {
            q->neighbors.emplace_back();
        }

        for (int lc = l; lc > l_new; lc--) {
            w = search_layer(q, ep, 1, lc);
            ep = w.top().second; // ep = nearest element from W to q
        }

        for (int lc = std::min(l, l_new); lc >= 0; lc--) {
            w = search_layer(q, ep, ef_construction, lc);

            std::vector<Node *> neighbors;
            if (select_neighbors_mode == "simple") {
                neighbors = select_neighbors_simple(w, m);
            } else if (select_neighbors_mode == "heuristic") {
                neighbors = select_neighbors_heuristic(q, w, m, lc, true, true);
            } else {
                throw std::runtime_error("select_neighbors_mode should be simple/heuristic");
            }

            // add bidirectional connections from neighbors to q at layer lc
            for (Node *e: neighbors) {
                e->neighbors[lc].push_back(q);
                q->neighbors[lc].push_back(e);
            }

            // shrink connections if needed
            for (Node *e: neighbors) {
                // if lc = 0 then m_max = m_max_0
                int m_effective = lc == 0 ? m_max_0 : m_max;

                std::vector<Node *> e_conn = e->neighbors[lc];
                if (e_conn.size() > m_effective) // shrink connections if needed
                {
                    std::vector<Node *> e_new_conn;
                    if (select_neighbors_mode == "simple") {
                        e_new_conn = select_neighbors_simple(e, e_conn, m_effective);
                    } else if (select_neighbors_mode == "heuristic") {
                        e_new_conn = select_neighbors_heuristic(e, e_conn, m_effective, lc, true, true);
                    } else {
                        throw std::runtime_error("select_neighbors_mode should be simple/heuristic");
                    }
                    e->neighbors[lc] = e_new_conn; // set neighborhood(e) at layer lc to e_new_conn
                }
            }
            ep = w.top().second;
        }
        if (l_new > l) {
            this->enter_point = q;
        }
    }

    std::priority_queue<std::pair<float, Node *> > search_layer(Node *q, Node *ep, int ef, int lc) {
        float d = dist_l2(&(ep->data), &(q->data));
        std::unordered_set<Node *> v{ep};                          // set of visited elements
        std::priority_queue<std::pair<float, Node *> > candidates; // set of candidates
        std::priority_queue<std::pair<float, Node *> > w;          // dynamic list of found nearest neighbors
        candidates.emplace(-d, ep);
        w.emplace(d, ep);

        while (!candidates.empty()) {
            Node *c = candidates.top().second; // extract nearest element from c to q
            float c_dist = candidates.top().first;
            candidates.pop();
            Node *f = w.top().second; // get furthest element from w to q
            float f_dist = w.top().first;
            if (-c_dist > f_dist) {
                break;
            }
            for (Node *e: c->neighbors[lc]) {
                if (v.find(e) == v.end()) {
                    v.emplace(e);
                    f = w.top().second;
                    float distance_e_q = dist_l2(&(e->data), &(q->data));
                    float distance_f_q = dist_l2(&(f->data), &(q->data));
                    if (distance_e_q < distance_f_q || w.size() < ef) {
                        candidates.emplace(-distance_e_q, e);
                        w.emplace(distance_e_q, e);
                        if (w.size() > ef) {
                            w.pop();
                        }
                    }
                }
            }
        }
        std::priority_queue<std::pair<float, Node *> > min_w;
        while (!w.empty()) {
            min_w.emplace(-w.top().first, w.top().second);
            w.pop();
        }
        return min_w;
    }

    std::vector<Node *> select_neighbors_simple(std::priority_queue<std::pair<float, Node *> > c, int m) {
        std::vector<Node *> neighbors;
        while (neighbors.size() < m && !c.empty()) {
            neighbors.push_back(c.top().second);
            c.pop();
        }
        return neighbors;
    }

    std::vector<Node *> select_neighbors_simple(Node *q, const std::vector<Node *> &c, int m) {
        std::priority_queue<std::pair<float, Node *> > w;
        for (Node *e: c) {
            w.emplace(dist_l2(&(e->data), &(q->data)), e);
            if (w.size() > m) {
                w.pop();
            }
        }
        return select_neighbors_simple(w, m);
    }


    std::vector<Node *> select_neighbors_heuristic(Node *q, std::priority_queue<std::pair<float, Node *>> c,
                                                   int m, int lc, bool extend_candidates,
                                                   bool keep_pruned_connections) {
        std::vector<Node *> v;
        while (!c.empty()) {
            v.push_back(c.top().second);
            c.pop();
        }
        return select_neighbors_heuristic(q, v, m, lc, extend_candidates, keep_pruned_connections);
    }

    std::vector<Node *> select_neighbors_heuristic(Node *q, const std::vector<Node *> &c,
                                                   int m, int lc, bool extend_candidates,
                                                   bool keep_pruned_connections) {
        std::priority_queue<std::pair<float, Node *>> r;  // (max heap)
        std::priority_queue<std::pair<float, Node *>> w;  // working queue for the candidates (min_heap)
        std::unordered_set<Node *> w_set; // this is to help check if e_adj is in w

        for (Node *n: c) {
            w.emplace(-dist_l2(&(q->data), &(n->data)), n);
            w_set.emplace(n);
        }

        if (extend_candidates) {
            for (Node *e: c) {
                for (Node *e_adj: (e->neighbors)[lc]) {
                    if (w_set.find(e_adj) == w_set.end()) {
                        w.emplace(-dist_l2(&(q->data), &(e_adj->data)), e_adj);
                        w_set.emplace(e_adj);
                    }
                }
            }
        }

        std::priority_queue<std::pair<float, Node *>> w_d; // queue for the discarded candidates
        while (!w.empty() && r.size() < m) {
            Node *e = w.top().second;
            float distance_e_q = w.top().first;
            w.pop();
            if (r.empty() || distance_e_q < r.top().first) {
                r.emplace(distance_e_q, e);
            } else {
                w_d.emplace(-distance_e_q, e);
            }
            if (keep_pruned_connections) { // add some of the discarded connections from w_d
                while (!w_d.empty() && r.size() < m) {
                    r.push(w_d.top());
                    w_d.pop();
                }
            }
        }

        // return r: convert r from pq to vector<Node *>
        std::vector<Node *> final;
        while (!r.empty()) {
            final.push_back(r.top().second);
            r.pop();
        }
        return final;
    }


    std::vector<std::vector<int> > knn_search(Node *q, int k, int ef) {
        std::priority_queue<std::pair<float, Node *> > w; // set for the current nearest elements
        Node *ep = this->enter_point;                     // get enter point for hnsw
        int l = ep->level;                                // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer(q, ep, 1, lc);
            ep = w.top().second;
        }
        w = search_layer(q, ep, ef, 0);

        std::vector<std::vector<int> > result;
        while (!w.empty() && result.size() < k) {
            result.push_back(w.top().second->data);
            w.pop();
        }
        return result; // return K nearest elements from W to q
    }

    std::vector<std::vector<int> > knn_search_brute_force(const Node *q, int k) {
        return knn_search_brute_force(q->data, k);
    }

    std::vector<std::vector<int> > knn_search_brute_force(const std::vector<int> &q, int k) {
        std::priority_queue<std::pair<float, std::vector<int> > > heap;
        for (const auto &i: data) {
            float dist = dist_l2(&i, &q);
            heap.emplace(dist, i);
            if (heap.size() > k) {
                heap.pop();
            }
        }
        std::vector<std::vector<int> > result;
        while (!heap.empty()) {
            result.push_back(heap.top().second);
            heap.pop();
        }
        return result;
    }
};

void load_fvecs_data(const char *filename,
                     std::vector<std::vector<int> > &results, unsigned &num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *) &dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    num = (unsigned) (fsize / (dim + 1) / 4);

    // initialize results
    results.resize(num);
    for (unsigned i = 0; i < num; i++)
        results[i].resize(dim);

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        float tmp[dim];
        in.read((char *) tmp, dim * 4);
        for (unsigned j = 0; j < dim; j++) {
            results[i][j] = (float) tmp[j];
        }
    }
    in.close();
}

float calculate_recall(const std::vector<std::vector<int> > &sample, const std::vector<std::vector<int> > &base) {
    struct hashFunction {
        size_t operator()(const std::vector<int>
                          &myVector) const {
            std::hash<int> hasher;
            size_t answer = 0;

            for (int i: myVector) {
                answer ^= hasher(i) + 0x9e3779b9 +
                          (answer << 6) + (answer >> 2);
            }
            return answer;
        }
    };
    int hit = 0;
    std::unordered_set < std::vector<int>, hashFunction > s;
    for (std::vector<int> v: base) {
        s.insert(v);
    }

    for (std::vector v: sample) {
        if (s.find(v) != s.end()) {
            hit++;
        }
    }
    return (float) hit / base.size();
}

int main(int argc, char **argv) {
    srand(42);
    // load dataset
    std::vector<std::vector<int> > base_load;
    std::vector<std::vector<int> > query_load;
    unsigned dim, num;
    load_fvecs_data("./siftsmall/siftsmall_base.fvecs", base_load, num, dim);

    std::cout << "result_num：" << num << std::endl
              << "result dimension：" << dim << std::endl;

    load_fvecs_data("./siftsmall/siftsmall_query.fvecs", query_load, num, dim);
    std::cout << "query_num：" << num << std::endl
              << "query dimension：" << dim << std::endl;

    // initialize graph
    auto start = std::chrono::high_resolution_clock::now();
    HNSW hnsw = HNSW();
    hnsw.build_graph(base_load);
    hnsw.print_graph_parameters();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "total time for building graph: " << (float) duration.count() / 1000 << std::endl;
    std::cout << "total distance count for building graph: " << hnsw.get_distance_calculation_count() << std::endl;

    // query
    start = std::chrono::high_resolution_clock::now();
    hnsw.set_distance_calculation_count(0);
    std::vector<std::vector<std::vector<int>>> query_result;
    for (std::vector<int> &v: query_load) {
        Node *query_node = new Node(v, std::vector<std::vector<Node *> >(), 0);
        query_result.push_back(hnsw.knn_search(query_node, 10, 20));
    }
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "total time for query: " << (float) duration.count() / 1000 << std::endl;
    std::cout << "total distance count for query: " << hnsw.get_distance_calculation_count() << std::endl;

    // calculate recall
    std::vector<float> total_recall;
    for (int i = 0; i < query_load.size(); i++) {
        total_recall.push_back(calculate_recall(query_result[i], hnsw.knn_search_brute_force(query_load[i], 10)));
    }
    float avg = std::accumulate(total_recall.begin(), total_recall.end(), 0.0) / total_recall.size();
    std::cout << "recall: " << avg << std::endl;

    return 0;
}