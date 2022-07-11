#include <iostream>
#include <utility>
#include <vector>
#include <math.h>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <iterator>
#include <fstream>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <unordered_map>


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

size_t hashFunc(const vector<float>& vec){
    hash<int> hasher;
    size_t res = 0;
    for (int i: vec) {
        res ^= hasher(i) + 0x9a3779c9 +
               (res << 4) + (res >> 3);
    }
    return res;
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
    int level;

    Node(const std::vector<float> &d, const std::vector<std::vector<Node *> > &n, int l) {
        data = d;
        neighbors = n;
        level = l;
    }
};

class HNSW {
private:
    std::vector<std::vector<float> > data;
    Node *enter_point = nullptr;


    // hyper parameters
    int m = 10;                                   // number of neighbors to connect in algo1
    int m_max = 20;                               // limit maximum number of neighbors in algo1
    int m_max_0 = 20;                             // limit maximum number of neighbors at layer0 in algo1
    int ef_construction = 40;                     // size of dynamic candidate list
    float ml = 1.0;                               // normalization factor for level generation
    unsigned long long int distance_calculation_count = 0;           // count number of calling distance function
    std::string select_neighbors_mode = "simple"; // select which select neighbor algorithm to use

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
    unordered_map<size_t , Node*> umap;
    std::vector<std::vector<Node *> > graph;
    std::vector<float> report_neighbor_connection() {
        std::vector<float> connectiveness;
        for (int l = 0; l < graph.size(); l++) {
            float connection_level = 0;
            for (Node * n : graph[l]) {
                auto closest_neighbors = this->knn_search_brute_force(n, graph[l], n->neighbors[l].size());
                std::vector<std::vector<float> > connected_neighbors;
                for (Node *ne : n->neighbors[l]) {
                    connected_neighbors.push_back(ne->data);
                }
                connection_level += calculate_recall(connected_neighbors, closest_neighbors);
            }
            connectiveness.push_back(connection_level / graph[l].size());
        }
        return connectiveness;
    }

    HNSW(int m, int m_max, int m_max_0, int ef_construction, float ml, std::string select_neighbors_mode) {
        srand(42);
        this->m = m;
        this->m_max = m_max;
        this->m_max_0 = m_max_0;
        this->ef_construction = ef_construction;
        this->ml = ml;
        this->select_neighbors_mode = select_neighbors_mode;
    }

    unsigned long long int get_distance_calculation_count() const {
        return distance_calculation_count;
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

        if (curr == total) {
            std::cout << std::endl;
        }
    }


    void build_graph(HNSW& hnsw, const std::vector<std::vector<float> > &input) {
        data = input;
        std::cout << "building graph" << std::endl;

        for (int i = 0; i < input.size(); i++) {
            // for (const std::vector<float> &i: input) {
            Node *node = new Node(input[i], std::vector<std::vector<Node *> >(), 0);
            hnsw.umap.insert({hashFunc(node->data), node});

            // special case: the first node has no enter point to insert
            if (enter_point == nullptr) {
                enter_point = node;
                node->neighbors.resize(1);
                graph.resize(1);
                graph[0].push_back(node);
                continue;
            }

            insert(hnsw, node, m, m_max, m_max_0, ef_construction, ml);

            // add new node to specific layer of graph
            while (graph.size() <= node->level) {
                graph.emplace_back();
            }
            for (int l = 0; l <= node->level; l++) {
                graph[l].push_back(node);
            }

            log_progress(i + 1, input.size());
        }

        // insert the nearest neighbor in layer 0
        for (int l = 0; l < graph.size(); l++) {
            for (Node*& nod: hnsw.graph[l]){
                int m_effective = l == 0 ? this->m_max_0 : this->m_max;
                vector<vector<float>> temp = knn_search_brute_force(nod, graph[l], m_effective);
                std::vector<Node *> new_neighbors;
                for (auto& vec: temp){
                    new_neighbors.push_back(hnsw.umap[hashFunc(vec)]);
                }
                nod->neighbors[l] = new_neighbors;
            }
        }
    }

    void insert(HNSW& hnsw, Node *q, int m, int m_max, int m_max_0, int ef_construction, float ml) {
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
        std::priority_queue<std::pair<float, Node *> > r;     // (max heap)
        std::priority_queue<std::pair<float, Node *> > w = c; // working queue for the candidates (min_heap)
        std::unordered_set<Node *> w_set;                     // this is to help check if e_adj is in w
        priority_queue<std::pair<float, Node *> > tempC1 = c;
        priority_queue<std::pair<float, Node *> > tempC2 = c;
        while (!tempC1.empty()) {
            w_set.emplace(tempC1.top().second);
            tempC1.pop();
        }

        if (extend_candidates) {
            while (!tempC2.empty()) {
                Node *e = tempC2.top().second;
                for (Node *e_adj: (e->neighbors)[lc]) {
                    if (w_set.find(e_adj) == w_set.end()) {
                        w.emplace(-dist_l2(&(q->data), &(e_adj->data)), e_adj);
                        w_set.emplace(e_adj);
                    }
                }
                tempC2.pop();
            }
        }

        std::priority_queue<std::pair<float, Node *> > w_d; // queue for the discarded candidates
        while (!w.empty() && r.size() < m) {
            Node *e = w.top().second;
            float distance_e_q = w.top().first;
            pair<float, Node*> cur_top = w.top();
            w.pop();
//            if (r.empty() || distance_e_q < r.top().first) {
//                r.emplace(distance_e_q, e);
//            } else {
//                w_d.emplace(-distance_e_q, e);
//            }
            bool good = true;
            priority_queue<pair<float, Node*>> temp_r = r;
            while (!temp_r.empty()){
                Node* top = temp_r.top().second;
                temp_r.pop();
                if (dist_l2(&top->data, &e->data) < distance_e_q){
                    good = false;
                    break;
                }
            }
            if (r.empty() || good){
                r.push(cur_top);
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
            final.emplace_back(r.top().second);
            r.pop();
        }
        return final;
    }

    std::vector<Node *> select_neighbors_heuristic(Node *q, const std::vector<Node *> &c,
                                                   int m, int lc, bool extend_candidates,
                                                   bool keep_pruned_connections) {
        std::priority_queue<std::pair<float, Node *> > r; // (max heap)
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
            final.emplace_back(r.top().second);
            r.pop();
        }
        return final;
    }

    std::vector<std::vector<float> > knn_search(Node *q, int k, int ef) {
        std::priority_queue<std::pair<float, Node *> > w; // set for the current nearest elements
        Node *ep = this->enter_point;                     // get enter point for hnsw
        int l = ep->level;                                // top level for hnsw
        for (int lc = l; lc > 0; lc--) {
            w = search_layer(q, ep, 1, lc);
            ep = w.top().second;
        }
        w = search_layer(q, ep, ef, 0);

        std::vector<std::vector<float> > result;
        while (!w.empty() && result.size() < k) {
            result.emplace_back(w.top().second->data);
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


void
build_graph_and_query(const std::vector<std::vector<float> > &base_load,
                      const std::vector<std::vector<float> > &query_load,
                      const std::vector<std::vector<float> > &ground_truth_load, std::string file_name, int m,
                      int m_max,
                      int m_max_0, int ef_construction, float ml, std::string select_neighbors_mode) {
    // initialize graph
    auto start = std::chrono::high_resolution_clock::now();
    HNSW hnsw = HNSW(m, m_max, m_max_0, ef_construction, ml, select_neighbors_mode);
    hnsw.build_graph(hnsw, base_load);
    hnsw.print_graph_parameters();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(end - start);
    float build_time = (float) duration.count();
    auto build_count = hnsw.get_distance_calculation_count();
    std::cout << "total time for building graph: " << build_time / 1000 << std::endl;
    std::cout << "total distance count for building graph: " << build_count << std::endl;

    // query
    start = std::chrono::high_resolution_clock::now();
    hnsw.set_distance_calculation_count(0);
    std::vector<std::vector<std::vector<float> > > query_result;
    for (const std::vector<float> &v: query_load) {
        Node *query_node = new Node(v, std::vector<std::vector<Node *> >(), 0);
        query_result.emplace_back(hnsw.knn_search(query_node, 100, 100));
    }
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::milliseconds>(end - start);
    float query_time = (float) duration.count();
    auto query_count = hnsw.get_distance_calculation_count();
    std::cout << "total time for query: " << query_time / 1000 << std::endl;
    std::cout << "total distance count for query: " << query_count << std::endl;

    // calculate recall
    std::vector<float> total_recall;
    for (int i = 0; i < query_load.size(); i++) {
        if (ground_truth_load.size() != 0) {
            total_recall.emplace_back(calculate_recall(query_result[i], base_load, ground_truth_load[i]));
        } else {
            total_recall.emplace_back(
                    calculate_recall(query_result[i], hnsw.knn_search_brute_force(query_load[i], base_load, 100)));
        }
    }
    float avg_recall = std::accumulate(total_recall.begin(), total_recall.end(), 0.0) / total_recall.size();
    std::cout << "recall: " << avg_recall << std::endl;

    // report neighbor connection
    std::vector<float> connectiveness = hnsw.report_neighbor_connection();
    std::string connection_accuracy;
    for (float f : connectiveness) {
        connection_accuracy += std::to_string(f) + " ";
    }

    // write to csv file
    std::fstream file(file_name, std::ios_base::app);
    file << m << ", " << m_max << ", " << m_max_0 << ", " << ef_construction << ", " << ml << ", "
         << select_neighbors_mode << ", "
         << build_time << ", " << query_time << ", " << build_count << ", " << query_count << ", " << avg_recall << ", "
         << connection_accuracy << "\n";
    file.close();
}

vector<float> split(const std::string &str) {
    std::string temp = "";
    std::vector<float> res;
    int count = 0;
    for (int i = 0; i < str.length(); i++) {
        if (str[i] == ' ' || i == str.length() - 1) {
            if (i == str.length() - 1) {
                temp += str[i];
            }
            if (count >= 1) {
                res.push_back(stof(temp));
            }
            count++;
            temp = "";
        } else {
            temp += str[i];
        }
    }
    return res;
}



int main(int argc, char **argv) {
    srand(42);
    // load dataset
    std::vector<std::vector<float> > base_load;
    std::vector<std::vector<float> > query_load;
    std::vector<std::vector<float> > ground_truth_load;
    unsigned dim1, num1;
    unsigned dim2, num2;
    unsigned dim3, num3;
    load_fvecs_data("siftsmall/siftsmall_base.fvecs", base_load, num1, dim1);
    load_fvecs_data("siftsmall/siftsmall_query.fvecs", query_load, num2, dim2);
    load_ivecs_data("siftsmall/siftsmall_groundtruth.ivecs", ground_truth_load, num3, dim3);
    std::cout << "base_num：" << num1 << std::endl
              << "base dimension：" << dim1 << std::endl;
    std::cout << "query_num：" << num2 << std::endl
              << "query dimension：" << dim2 << std::endl;

    // glove
//    std::string filename = "glove.twitter.27B/glove.twitter.27B.25d.txt";
//    std::ifstream fd;
//    fd.open(filename);
//    std::string temp;
//    while (getline(fd, temp)) {
//        base_load.push_back(split(temp));
//    }
//    fd.open("glove.twitter.27B.25d.ground_truth.txt");
//    while (getline(fd, temp)) {
//        ground_truth_load.push_back(split(temp));
//    }
//    for (int i = base_load.size() - 1; i > base_load.size() - 10001; i--) {
//        query_load.emplace_back(base_load[i]);
//        //base_load.pop_back();
//    }

    std::cout << "base_num：" << base_load.size() << std::endl
              << "base dimension：" << base_load[0].size() << std::endl;

    std::cout << "query_num：" << query_load.size() << std::endl
              << "query dimension：" << query_load[0].size() << std::endl;

    // prepare csv file to write
    std::string file_name = "test.csv";
    std::fstream output_file(file_name, std::ios_base::out);
    output_file << "m, m_max, m_max_0, ef_construction, ml, select_neighbor_mode, "
                << "total_time_for_building_graph, total_time_for_query, total_distance_count_for_building_graph, total_distance_count_for_query, "
                << "recall\n";
    output_file.close();

    for (std::string select_neighbors_mode: {"simple", "heuristic"}) {
        for (int m = 5; m < 25; m += 5) {
            for (int m_max = 5; m_max < 45; m_max += 5) {
                for (int m_max_0 = 5; m_max_0 < 50; m_max_0 += 5) {
                    for (int ef_construction = 10; ef_construction < 20; ef_construction += 20) {
                        build_graph_and_query(base_load, query_load, ground_truth_load, file_name, m, m_max, m_max_0,
                                              ef_construction, 1.0, select_neighbors_mode);
                    }
                }
            }
        }
    }

    return 0;
}
