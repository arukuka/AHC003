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
        {"update_ratio", 0.559537888101038},
        {"update_decrease_ratio_by_time", 0.8798946551390772},
        {"update_decrease_ratio_by_num", 1},
        {"update_weak_ratio", 0.19709048518341163},
        {"update_weak_decrease_ratio_by_time",  0.997192165269988},
        {"update_weak_decrease_ratio_by_num", 1},
        {"update_strong_weak_ratio", 1},
        {"width_ratio", 0.3440998384706083},
        {"width_decrease_ratio_by_time", 0.9990981780982433},
        {"width_decrease_ratio_by_step", 1}
    };

    void update()
    {
        dict["update_ratio"]      *= dict["update_decrease_ratio_by_time"];
        dict["update_weak_ratio"] *= dict["update_weak_decrease_ratio_by_time"];
        dict["width_ratio"]       *= dict["width_decrease_ratio_by_time"];
    }

    double operator[](std::string s)
    {
        return dict.at(s);
    }
};
Parameters parameters;

struct Probability
{
    double distance;
    double min;
    double max;
    int update_count;
    int update_weak_count;

    template<typename RandomEngine>
    void generate(RandomEngine& rnd)
    {
        distance = std::uniform_real_distribution<>(min, max)(rnd);
    }

    void update(double next)
    {
        const double ratio = parameters["update_ratio"]
                * std::pow(
                    parameters["update_decrease_ratio_by_num"]
                ,
                      update_count      *      parameters["update_strong_weak_ratio"]
                    + update_weak_count * (1 - parameters["update_strong_weak_ratio"]));
        const double rev = 1 - ratio;
        min = min * rev + next * ratio;
        max = max * rev + next * ratio;
        const double tmp = std::min(min, max);
        max = std::max(min, max) + 1e-9;
        min = tmp;
        ++update_count;
    }

    void update_weak(double next, int step)
    {
        const double ratio =
                parameters["update_weak_ratio"]
                    * std::pow(
                        parameters["update_weak_decrease_ratio_by_num"]
                    ,
                          update_count      *      parameters["update_strong_weak_ratio"]
                        + update_weak_count * (1 - parameters["update_strong_weak_ratio"])
                    )
                    * std::pow(parameters["width_decrease_ratio_by_step"], step);
        const double rev = 1 - ratio;
        min = min * rev + next * ratio;
        max = max * rev + next * ratio;
        const double tmp = std::min(min, max);
        max = std::max(min, max) + 1e-9;
        min = tmp;
        ++update_weak_count;
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
                const int width = static_cast<int>(parameters["width_ratio"] * N + 0.5);

                for (int w = 0; w < width; ++w)
                {
                    const int nr = r + OFS[foward_dir].r;
                    const int nc = c + OFS[foward_dir].c;
                    if (is_out(nr, nc))
                    {
                        break;
                    }
                    grids[ r][ c].adj[  foward_dir].update_weak(average, w);
                    grids[nr][nc].adj[backward_dir].update_weak(average, w);
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
        "update_ratio",
        "update_decrease_ratio_by_time",
        "update_decrease_ratio_by_num",
        "update_weak_ratio",
        "update_weak_decrease_ratio_by_time",
        "update_weak_decrease_ratio_by_num",
        "update_strong_weak_ratio",
        "width_ratio",
        "width_decrease_ratio_by_time",
        "width_decrease_ratio_by_step",
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
                v.min = MIN_DIST;
                v.max = MAX_DIST;
                v.update_count = v.update_weak_count = 0;
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
