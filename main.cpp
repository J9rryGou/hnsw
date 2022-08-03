#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <iterator>
#include <fstream>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

float calculate_recall(const std::vector<std::vector<float> > &sample, const std::vector<std::vector<float> > &base) {
    struct hashFunction {
        size_t operator()(const std::vector<float>
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
    std::unordered_set < std::vector<float>, hashFunction > s;
    for (std::vector<float> v: base) {
        s.insert(v);
    }

    for (std::vector<float> v: sample) {
        if (s.find(v) != s.end()) {
            hit++;
        }
    }
    return (float) hit / base.size();
}

float
calculate_recall(const std::vector<std::vector<float> > &sample, const std::vector<std::vector<float> > &base_load,
                 const vector<float> &index) {
    std::vector<std::vector<float> > base;
    for (int i = 0; i < index.size(); i++) {
        base.push_back(base_load[index[i]]);
    }
    return calculate_recall(sample, base);
}


class Node {
public:
    std::vector<float> data;
    std::vector<std::vector<Node *> > neighbors;
    Node *parent;
    std::vector<int> next;

    Node(const std::vector<float> &d, const std::vector<std::vector<Node *> > &n) {
        this->data = d;
        this->neighbors = n;
        this->parent = nullptr;
    }
};

class HNSW {
private:
    // hyper parameters
    int m;                                   // number of neighbors to connect in algo1
    int m_max;                               // limit maximum number of neighbors in algo1
    int m_max_0;                             // limit maximum number of neighbors at layer0 in algo1
    int ef_construction;                     // size of dynamic candidate list
    float ml;                                // normalization factor for level generation
    std::string select_neighbors_mode;       // select which select neighbor algorithm to use

    // statistics
    unsigned long long int distance_calculation_count;           // count number of calling distance function
    int level_one_hit_count;

    float dist_l2(const std::vector<float> *v1, const std::vector<float> *v2) {
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
    Node *enter_point = nullptr;
    std::map<Node *, std::map<Node *, std::map<int, int> > > edge_map;

    HNSW(int m, int m_max, int m_max_0, int ef_construction, float ml, const std::string &select_neighbors_mode) {
        srand(42);
        this->m = m;
        this->m_max = m_max;
        this->m_max_0 = m_max_0;
        this->ef_construction = ef_construction;
        this->ml = ml;
        this->select_neighbors_mode = select_neighbors_mode;
        this->distance_calculation_count = 0;
        this->level_one_hit_count = 0;
        this->enter_point = nullptr;
    }

    std::tuple<int, int, int, int, float, std::string> get_graph_parameters() {
        return std::make_tuple(m, m_max, m_max_0, ef_construction, ml, select_neighbors_mode);
    }

    unsigned long long int get_distance_calculation_count() const {
        return distance_calculation_count;
    }

    int get_level_one_hit_count() const {
        return level_one_hit_count;
    }

    void set_distance_calculation_count(unsigned long long int set_count) {
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

        if (curr >= total) {
            std::cout << std::endl;
        }
    }


    void build_graph(const std::vector<std::vector<float> > &input) {
        std::cout << "building graph" << std::endl;

        for (int i = 0; i < input.size(); i++) {
            Node *node = new Node(input[i], std::vector<std::vector<Node *> >());

            // special case: the first node has no enter point to insert
            if (enter_point == nullptr) {
                enter_point = node;
                node->neighbors.resize(1);
                node->next.resize(1);
                continue;
            }
            insert(node, m, m_max, m_max_0, ef_construction, ml);
            log_progress(i + 1, input.size());
        }
    }

    void insert(Node *q, int m, int m_max, int m_max_0, int ef_construction, float ml) {
        std::priority_queue<std::pair<float, Node *> > w;
        Node *ep = this->enter_point;
        int l = ep->neighbors.size() - 1;
        int l_new = floor(-log((float) rand() / (RAND_MAX + 1.0)) * ml);

        // update fields of node to level l_new
        q->neighbors.resize(l_new + 1);
        q->next.resize(l_new + 1);

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
                neighbors = select_neighbors_heuristic(q, w, m, lc, true, false);
            } else {
                throw std::runtime_error("select_neighbors_mode should be simple/heuristic");
            }

            // add bidirectional connections from neighbors to q at layer lc
            for (Node *e: neighbors) {
                e->neighbors[lc].emplace_back(q);
                q->neighbors[lc].emplace_back(e);
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
                        e_new_conn = select_neighbors_heuristic(e, e_conn, m_effective, lc, true, false);
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
                    // record parent
                    e->parent = c;
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

    std::priority_queue<std::pair<float, Node *> > search_layer_new(Node *q, Node *ep, int ef, int lc) {
        float d = dist_l2(&(ep->data), &(q->data));
        std::unordered_set<Node *> v{ep};                          // set of visited elements
        std::priority_queue<std::pair<float, Node *> > candidates; // set of candidates
        std::priority_queue<std::pair<float, Node *> > w;          // dynamic list of found nearest neighbors
        candidates.emplace(-d, ep);
        w.emplace(d, ep);
        ep->next[lc] = 0;

        while (!candidates.empty()) {
            Node *c = candidates.top().second; // extract nearest element from c to q
            float c_dist = candidates.top().first;
            Node *f = w.top().second; // get furthest element from w to q
            float f_dist = w.top().first;
            if (-c_dist > f_dist) {
                break;
            }

            if (c->next[lc] >= c->neighbors[lc].size()) {
                candidates.pop();
            } else {
                Node *e = c->neighbors[lc][c->next[lc]];
                c->next[lc]++;
                if (v.find(e) == v.end()) {
                    v.emplace(e);
                    e->parent = c; // record parent
                    f = w.top().second;
                    float distance_e_q = dist_l2(&(e->data), &(q->data));
                    float distance_f_q = dist_l2(&(f->data), &(q->data));
                    if (distance_e_q < distance_f_q || w.size() < ef) {
                        e->next[lc] = 0;
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
            neighbors.emplace_back(c.top().second);
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

    std::vector<Node *> select_neighbors_heuristic(Node *q, std::priority_queue<std::pair<float, Node *> > c,
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
        std::vector<Node *> r; // (max heap)
        std::priority_queue<std::pair<float, Node *> > w; // working queue for the candidates (min_heap)
        std::unordered_set<Node *> w_set;                 // this is to help check if e_adj is in w

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

        std::priority_queue<std::pair<float, Node *> > w_d; // queue for the discarded candidates
        while (!w.empty() && r.size() < m) {
            Node *e = w.top().second;
            float distance_e_q = w.top().first;
            w.pop();
            bool good = true;
            for (Node * rr : r) {
                if (dist_l2(&rr->data, &e->data) < distance_e_q){
                    good = false;
                    break;
                }
            }
            if (r.empty() || good){
                r.push_back(e);
            } else {
                w_d.emplace(-distance_e_q, e);
            }
            if (keep_pruned_connections) { // add some of the discarded connections from w_d
                while (!w_d.empty() && r.size() < m) {
                    r.push_back(w_d.top().second);
                    w_d.pop();
                }
            }
        }

        return r;
    }


    std::vector<std::vector<float> > knn_search_new(Node *q, int k, int ef) {
        std::priority_queue<std::pair<float, Node *> > w; // set for the current nearest elements
        Node *ep = this->enter_point;                     // get enter point for hnsw
        int l = ep->neighbors.size() - 1;                 // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer_new(q, ep, 1, lc);
            ep = w.top().second;
        }

        w = search_layer_new(q, ep, ef, 0);

        std::vector<std::vector<float> > result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second->data);
            w.pop();
        }
        return result; // return K nearest elements from W to q
    }

    std::vector<std::vector<float> > knn_search(Node *q, int k, int ef) {
        std::priority_queue<std::pair<float, Node *> > w; // set for the current nearest elements
        Node *ep = this->enter_point;                     // get enter point for hnsw
        int l = ep->neighbors.size() - 1;                 // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer(q, ep, 1, lc);
            Node *p  = w.top().second;
            if (p == ep) {
                this->edge_map[p][p][lc]++;
            }
            while (p != ep) {
                this->edge_map[p->parent][p][lc]++;
                p = p->parent;
            }
            ep = w.top().second;
        }

        w = search_layer(q, ep, ef, 0);

        std::vector<std::vector<float> > result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second->data);
            Node *p  = w.top().second;
            if (p == ep) {
                this->edge_map[p][p][0]++;
            }
            while (p != ep) {
                this->edge_map[p->parent][p][0]++;
                p = p->parent;
            }
            w.pop();
        }
        return result; // return K nearest elements from W to q
    }

    std::vector<std::vector<float> >
    knn_search_brute_force(const Node *q, const std::vector<Node *> &base_data_nodes, int k) {
        std::vector<std::vector<float> > base_data;
        for (const Node * const n : base_data_nodes) {
            base_data.emplace_back(n->data);
        }
        return knn_search_brute_force(q->data, base_data, k);
    }

    std::vector<std::vector<float> >
    knn_search_brute_force(const std::vector<float> &q, const std::vector<std::vector<float> > &base_data, int k) {
        std::priority_queue<std::pair<float, std::vector<float> > > heap;
        for (const auto &i: base_data) {
            float dist = dist_l2(&i, &q);
            heap.emplace(dist, i);
            if (heap.size() > k) {
                heap.pop();
            }
        }
        std::vector<std::vector<float> > result;
        while (!heap.empty()) {
            result.emplace_back(heap.top().second);
            heap.pop();
        }
        return result;
    }
};

void load_fvecs_data(const char *filename,
                     std::vector<std::vector<float> > &results, unsigned &num, unsigned &dim) {
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

void load_ivecs_data(const char *filename,
                     std::vector<std::vector<float> > &results, unsigned &num, unsigned &dim) {
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
        int tmp[dim];
        in.read((char *) tmp, dim * 4);
        for (unsigned j = 0; j < dim; j++) {
            results[i][j] = (int) tmp[j];
        }
    }
    in.close();
}

void load_txt_data(const char* filename, std::vector<std::vector<float> > &results, unsigned &num, unsigned &dim) {
    std::ifstream fd(filename);
    if (!fd.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    std::string temp;
    while (getline(fd, temp)) {
        vector<float> f;
        // split the line by space
        std::istringstream iss(temp);
        std::string token;
        while (getline(iss, token, ' ')) {
            f.push_back(stof(token));
        }
        results.push_back(f);
    }

    num = results.size();
    dim = results[0].size();
    fd.close();
}


void
build_graph_and_query(const std::vector<std::vector<float> > &base_load,
                      const std::vector<std::vector<float> > &learn_load,
                      const std::vector<std::vector<float> > &query_load,
                      const std::vector<std::vector<float> > &groundtruth_load,
                      std::string file_name, HNSW &hnsw, int k, int ef_k) {
    // initialize graph
    hnsw.print_graph_parameters();
    auto start = std::chrono::high_resolution_clock::now();
    hnsw.build_graph(base_load);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(end - start);
    float build_time = (float) duration.count();
    auto build_count = hnsw.get_distance_calculation_count();
    std::cout << "total time for building graph: " << build_time / 1000 << std::endl;
    std::cout << "total distance count for building graph: " << build_count << std::endl;

    // learn
    std::cout << "querying learn vectors..." << std::endl;
    for (size_t i = 0; i < learn_load.size(); i++) {
        Node *query_node = new Node(learn_load[i], std::vector<std::vector<Node *>>());
        hnsw.knn_search(query_node, k, ef_k);
        delete query_node;
        hnsw.log_progress(i + 1, learn_load.size());
    }


    // base
    std::cout << "querying base vectors..." << std::endl;
    for (size_t i = 0; i < base_load.size(); i++) {
        Node *query_node = new Node(base_load[i], std::vector<std::vector<Node *>>());
        hnsw.knn_search(query_node, k, ef_k);
        delete query_node;
        hnsw.log_progress(i + 1, base_load.size());
    }

    //remove edges in hnsw.edge_map
//    for (const auto &i: hnsw.edge_map) {
//        for (const auto &j: i.second) {
//            for (const auto &k: j.second) {
//                if (hnsw.edge_map[i.first][j.first][k.first] < 15) {
//                    std::vector<Node *> &v = i.first->neighbors[k.first];
//                    v.erase(std::remove(v.begin(), v.end(), j.first), v.end());
//                }
//            }
//        }
//    }

    // sort all node neighbors by frequency count in edge_map
    std::cout << "sorting neighbors..." << std::endl;
    for (auto &i: hnsw.edge_map) {
        i.first->next = std::vector<int>(i.first->neighbors.size(), 0);
        for (int l = 0; l < i.first->neighbors.size(); l++) {
            std::sort(i.first->neighbors[l].begin(), i.first->neighbors[l].end(),
                    [&hnsw, &i, l](Node *a, Node *b) {
                        return hnsw.edge_map[i.first][a][l] > hnsw.edge_map[i.first][b][l];
                    });
        }
    }

    // query
    std::cout << "querying query vectors..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    hnsw.set_distance_calculation_count(0);
    std::vector<std::vector<std::vector<float> > > query_result;
    for (size_t i = 0; i < query_load.size(); i++) {
        Node *query_node = new Node(query_load[i], std::vector<std::vector<Node *> >());
        query_result.emplace_back(hnsw.knn_search_new(query_node, k, ef_k));
        delete query_node;
        hnsw.log_progress(i + 1, query_load.size());
    }

    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::milliseconds>(end - start);
    float query_time = (float) duration.count();
    auto query_count = hnsw.get_distance_calculation_count();
    std::cout << "total time for query: " << query_time / 1000 << std::endl;
    std::cout << "total distance count for query: " << query_count << std::endl;

    // frequency distribution
    // std::map<int, int> freq_dist;
    // for (int l = 0; l < hnsw.enter_point->neighbors.size(); l++) {
    //     std::cout << "level " << l << std::endl;
    //     int count = 0;
    //     for (Node *a : hnsw.edge_map.key_comp) {
    //         std::cout << "node " << a << ": ";
    //         std::cout <<hnsw.edge_map[a][a][l] << " | ";
    //         count += hnsw.edge_map[a][a][l];
    //         for (Node *b: a->neighbors[l]) {
    //             freq_dist[hnsw.edge_map[a][b][l]]++;
    //             std::cout << hnsw.edge_map[a][b][l] << " ";
    //             count += hnsw.edge_map[a][b][l];
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << "level " << l << " total count: " << count << std::endl << std::endl;
    // }
    // std::cout << "frequency distribution: " << std::endl;
    // for (const auto &i: freq_dist) {
    //     std::cout << i.first << ": " << i.second << std::endl;
    // }

    // calculate recall
    std::vector<float> total_recall;
    for (int i = 0; i < query_load.size(); i++) {
        if (groundtruth_load.size() != 0) {
            total_recall.emplace_back(calculate_recall(query_result[i], base_load, groundtruth_load[i]));
        } else {
            total_recall.emplace_back(
                    calculate_recall(query_result[i], hnsw.knn_search_brute_force(query_load[i], base_load, 100)));
        }
    }
    float avg_recall = std::accumulate(total_recall.begin(), total_recall.end(), 0.0) / total_recall.size();
    std::cout << "recall: " << avg_recall << std::endl;

    // write to csv file
    auto p = hnsw.get_graph_parameters();
    int m, m_max, m_max_0, ef_construction;
    float ml;
    std::string select_neighbors_mode;
    std::tie(m, m_max, m_max_0, ef_construction, ml, select_neighbors_mode) = p;
    file_name = "./result/hnsw_" + std::to_string(m) + "_" + std::to_string(m_max) + "_" + std::to_string(m_max_0) +
                "_" + std::to_string(ef_construction) + "_" + std::to_string(ml) + "_" + select_neighbors_mode +
                "_" + std::to_string(k) + "_" + std::to_string(ef_k) + ".csv";
    std::fstream file(file_name, std::ios_base::app);
    file << m << "," << m_max << "," << m_max_0 << "," << ef_construction << "," << ml << ","
         << select_neighbors_mode << ","
         << build_time << "," << query_time << "," << build_count << "," << query_count << "," << avg_recall << ","
         << k << "," << ef_k << "\n";
    file.close();
}


int main(int argc, char **argv) {
    srand(42);
    // load dataset
    std::vector<std::vector<float> > base_load;
    std::vector<std::vector<float> > query_load;
    std::vector<std::vector<float> > learn_load;
    std::vector<std::vector<float> > groundtruth_load;
    unsigned dim1, dim2, dim3, dim4;
    unsigned num1, num2, num3, num4;
    // load_fvecs_data("sift/sift_base.fvecs", base_load, num1, dim1);
    // load_fvecs_data("sift/sift_learn.fvecs", learn_load, num2, dim2);
    // load_fvecs_data("sift/sift_query.fvecs", query_load, num3, dim3);
    // load_ivecs_data("sift/sift_groundtruth.ivecs", groundtruth_load, num4, dim4);
//
    load_txt_data("glove.twitter.27B.25d/glove.twitter.27B.25d.base.txt", base_load, num1, dim1);
    load_txt_data("glove.twitter.27B.25d/glove.twitter.27B.25d.query.txt", query_load, num2, dim2);
    load_txt_data("glove.twitter.27B.25d/glove.twitter.27B.25d.learn.txt", learn_load, num3, dim3);
    load_txt_data("glove.twitter.27B.25d/glove.twitter.27B.25d.groundtruth.txt", groundtruth_load, num4, dim4);

    std::cout << "base_num：" << num1 << std::endl
              << "base dimension：" << dim1 << std::endl;
    std::cout << "learn_num：" << num2 << std::endl
              << "learn dimension：" << dim2 << std::endl;
    std::cout << "query_num：" << num3 << std::endl
              << "query dimension：" << dim3 << std::endl;
    std::cout << "groundtruth_num：" << num4 << std::endl
              << "groundtruth dimension：" << dim4 << std::endl;

    // prepare csv file to write
    std::string file_name = "test.csv";
    std::fstream output_file(file_name, std::ios_base::app);
    output_file << "m,m_max,m_max_0,ef_construction,ml,select_neighbor_mode,"
                << "total_time_for_building_graph,total_time_for_query,total_distance_count_for_building_graph,total_distance_count_for_query,"
                << "recall,k,ef_k\n";
    output_file.close();

    // for (std::string select_neighbors_mode: {"simple", "heuristic"}) {
    //     for (int m = 10; m < 25; m += 5) {
    //         for (int m_max = 10; m_max < 45; m_max += 5) {
    //             for (int m_max_0 = 10; m_max_0 < 50; m_max_0 += 5) {
    //                 for (int ef_construction = m; ef_construction < 120; ef_construction += 20) {
    //                     build_graph_and_query(base_load, query_load, groundtruth_load, file_name, m, m_max, m_max_0,
    //                                           ef_construction, 1.0, select_neighbors_mode, 100, 100);
    //                 }
    //             }
    //         }
    //     }
    // }
    HNSW hnsw = HNSW(16, 16, 32, 32, 1.0, "simple");
    build_graph_and_query(base_load, learn_load, query_load, groundtruth_load, file_name, hnsw, 100, 5000);

    return 0;
}
