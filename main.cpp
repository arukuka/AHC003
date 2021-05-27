#include <functional>
#include <iostream>
#include <limits>
#include <ostream>
#include <vector>
#include <array>
#include <cstring>
#include <queue>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <map>

#ifdef LOCAL
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/system/error_code.hpp>
#endif

constexpr int N = 30;
constexpr int MIN_DIST = 1000;
constexpr int MAX_DIST = 9000;
constexpr int NUM_QUERIS = 1000;
constexpr int MIN_DIFF = 100;
constexpr int MAX_DIFF = 2000;

struct Point
{
    int r, c;

    bool operator==(const Point& rhs) const
    {
        return r == rhs.r && c == rhs.c;
    }
    bool operator!=(const Point& rhs) const
    {
        return !this->operator==(rhs);
    }
};

std::ostream& operator<<(std::ostream& os, const Point& p)
{
    os << "(" << p.r << ", " << p.c << ")";
    return os;
}

std::array<Point, 4> OFS = {{
    {0, 1},
    {1, 0},
    {0, -1},
    {-1, 0}
}};

constexpr char CMD_DIR[5] = "RDLU";

struct Path
{
    Point start;
    double score;
    std::vector<int> dirs;

    operator std::string() const
    {
        std::stringstream ss;
        for (const int& dir : dirs)
        {
            ss << CMD_DIR[dir];
        }

        std::string ans = ss.str();

        return ans;
    }
};

struct UCB1
{
    double reward_average;
    int num_selected;
    int num_all;

    double calculate() const
    {
        double score = reward_average;
        score += std::sqrt(2.0 * std::log2(num_all) / static_cast<double>(num_selected));
        score = 1.0 / score;
        return score;
    }
};

struct Parameters
{
    std::map<std::string, double> dict = {
        {"width_ratio", 0.3440998384706083},
        {"width_decrease_ratio", 0.9990981780982433},
        {"use_analyze_thr", 0},
        {"default_value", 0},
        {"default_width", 0},
    };

    void update()
    {
        dict["width_ratio"]       *= dict["width_decrease_ratio"];
    }

    double operator[](const std::string& s) const
    {
        return dict.at(s);
    }

    int get_update_width() const
    {
        return static_cast<int>(dict.at("width_ratio") * N + 0.5);
    }
};
Parameters parameters;

struct Probability
{
    static constexpr int HIST_DIV = 500;
    static constexpr int HIST_NUM = MAX_DIST / HIST_DIV + 1;
    static constexpr int SEARCH_NUMS = 1;
    static constexpr int SEARCH_STEP = HIST_DIV / SEARCH_NUMS;

    double distance;
    double min;
    double max;
    int update_num[HIST_NUM];
    bool updated;

    void analyze()
    {
        int best_min = -1;
        int best_max = -1;
        int best_score = std::numeric_limits<int>::max();

        int sum = 0;
        for (int h = 0; h < HIST_NUM; ++h)
        {
            sum += update_num[h];
        }

        if (sum < parameters["use_analyze_thr"])
        {
            min = parameters["default_value"] - parameters["default_width"];
            min = parameters["default_value"] + parameters["default_width"];
        }

        for (int _diff = std::max(1, MIN_DIFF / HIST_DIV); _diff <= MAX_DIFF / HIST_DIV; ++_diff)
        {
            const int diff = _diff * HIST_DIV;
            for (int _min = (MIN_DIST + diff) / SEARCH_STEP; _min < (MAX_DIST - diff) / SEARCH_STEP; ++_min)
            {
                const int min = _min * SEARCH_STEP;
                const int max = min + diff;
                const double ave = static_cast<double>(sum) / _diff;
                int score = 0;
                for (int h = 0; h < HIST_NUM; ++h)
                {
                    const int val = h * HIST_DIV;
                    const int cnt = update_num[h];
                    const double d = min <= val && val <= max
                                   ? cnt - ave
                                   : cnt;

                    const int d2 = d * d;

                    score += d2;
                }

                if (score < best_score)
                {
                    best_score = score;
                    best_min = min;
                    best_max = max;
                }
            }
        }

        min = best_min;
        max = best_max;
    }

    template<typename RandomEngine>
    void generate(RandomEngine& rnd)
    {
        if (updated)
        {
            analyze();
            updated = false;
        }

        distance = std::uniform_real_distribution<>(min, max)(rnd);
    }

    void update(double next)
    {
        const int index = static_cast<int>((next + HIST_DIV / 2.0) / HIST_DIV);
        update_num[index]++;
        updated = true;
    }
};

struct Node
{
    int r, c;
    std::array<Probability, 4> adj;
};

std::array<std::array<Node, N>, N> grids;

bool is_out(const int r, const int c)
{
    return r < 0 || N <= r
           || c < 0 || N <= c;
}

bool is_out(const Point& p)
{
    return is_out(p.r, p.c);
}

int memo[N][N];

struct DijkstraNode
{
    Point cur;
    int dir;
    int length;

    bool operator<(const DijkstraNode& rhs) const
    {
        return length < rhs.length;
    }
    bool operator>(const DijkstraNode& rhs) const
    {
        return length > rhs.length;
    }
};

Path dijkstra(const Point& s, const Point& t)
{
    std::memset(memo, -1, sizeof(memo));

    DijkstraNode initial{s, 4, 0};
    std::priority_queue<
        DijkstraNode,
        std::vector<DijkstraNode>,
        std::greater<DijkstraNode>
    > queue;

    queue.push(initial);
    while (!queue.empty())
    {
        const DijkstraNode node = queue.top();
        queue.pop();

        if (memo[node.cur.r][node.cur.c] != -1)
        {
            continue;
        }
        memo[node.cur.r][node.cur.c] = node.dir;

        if (node.cur == t)
        {
            break;
        }

        for (int d = 0; d < 4; ++d)
        {
            DijkstraNode next{node.cur, d, node.length};
            next.cur.r += OFS[d].r;
            next.cur.c += OFS[d].c;
            if (is_out(next.cur))
            {
                continue;
            }
            next.length += grids[node.cur.r][node.cur.c].adj[d].distance;
            queue.push(next);
        }
    }

    std::vector<int> dirs;
    Point ite = t;
    while (ite != s)
    {
        const int d = memo[ite.r][ite.c];
        dirs.push_back(d);
        ite.r -= OFS[d].r;
        ite.c -= OFS[d].c;
    }

    std::reverse(dirs.begin(), dirs.end());

    Path ans{s, 0, dirs};

    return ans;
}

void update(Path& path, const int score)
{
    const int size = static_cast<int>(path.dirs.size());
    path.score = score;
    const double average = path.score / size;
    const double reward = (MAX_DIST - average) / (MAX_DIST - MIN_DIST);

    Point ite = path.start;
    for (const int& dir : path.dirs)
    {
        const int rev_dir = (dir + 2) % 4;
        Point next = ite;
        next.r += OFS[dir].r;
        next.c += OFS[dir].c;

        // UPDATE
        grids[ ite.r][ ite.c].adj[    dir].update(average);
        grids[next.r][next.c].adj[rev_dir].update(average);
        {
            // UPDATE WEAK
            const std::array<Point, 2> start_points = {
                ite,
                next
            };
            const std::array<int, 2> update_dirs = {
                rev_dir,
                dir
            };
            for (int i = 0; i < 2; ++i)
            {
                const Point sp = start_points[i];
                int r = sp.r;
                int c = sp.c;
                const int   foward_dir =  update_dirs[i];
                const int backward_dir = (update_dirs[i] + 2) % 4;
                const int width = parameters.get_update_width();

                for (int w = 0; w < width; ++w)
                {
                    const int nr = r + OFS[foward_dir].r;
                    const int nc = c + OFS[foward_dir].c;
                    if (is_out(nr, nc))
                    {
                        break;
                    }
                    grids[ r][ c].adj[  foward_dir].update(average);
                    grids[nr][nc].adj[backward_dir].update(average);
                    r = nr;
                    c = nc;
                }
            }
        }

        ite = next;
    }
}

#ifdef LOCAL
void load_params(const boost::filesystem::path params_path)
{
    namespace fs = boost::filesystem;

    boost::system::error_code error;
    const bool is_exists = fs::exists(params_path, error);
    if (!is_exists || error)
    {
        std::cerr << params_path << ": Not Found (" << error << ")" << std::endl;
    }

    namespace pt = boost::property_tree;

    pt::ptree property_tree;
    pt::read_json(fs::absolute(params_path).c_str(), property_tree);

    const std::vector<std::string> params_list = {
        "width_ratio",
        "width_decrease_ratio",
        "use_analyze_thr",
        "default_value",
        "default_width"
    };

    for (const auto& param_name : params_list)
    {
        parameters.dict[param_name] = property_tree.get<double>(param_name);
    }
}
#endif

int main(const int argc, const char * const * const argv)
{
#ifdef LOCAL
    {
        using namespace boost::program_options;

        options_description description;
        description.add_options()
            ("parameters,p", value<boost::filesystem::path>())
            ("help,h", "help")
            ;

        variables_map vm;
        store(parse_command_line(argc, argv, description), vm);
        notify(vm);

        if (vm.count("help"))
        {
            std::cerr << description << std::endl;
            return 0;
        }

        if (vm.count("parameters"))
        {
            load_params(vm["parameters"].as<boost::filesystem::path>());
        }
    }
#endif

    std::mt19937 random_egine(std::hash<std::string>()("arukuka"));

    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            for (auto& v : grids[r][c].adj)
            {
                v.updated = true;
                v.generate(random_egine);
            }
        }
    }

    for (int q = 0; q < NUM_QUERIS; ++q)
    {
        Point s;
        Point t;
        std::cin >> s.r >> s.c >> t.r >> t.c;

        auto path = dijkstra(s, t);

        std::cout << static_cast<std::string>(path) << std::endl;
        std::flush(std::cout);

        int score;
        std::cin >> score;

        update(path, score);

        parameters.update();
        for (int r = 0; r < N; ++r)
        {
            for (int c = 0; c < N; ++c)
            {
                for (auto& v : grids[r][c].adj)
                {
                    v.generate(random_egine);
                }
            }
        }
    }

    return 0;
}
