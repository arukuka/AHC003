#include <functional>
#include <iostream>
#include <ostream>
#include <vector>
#include <array>
#include <cstring>
#include <queue>
#include <sstream>
#include <string>
#include <algorithm>

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

struct Node
{
    int r, c;
    std::array<double, 4> adj;
    std::vector<Path> paths;
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
            next.length += grids[node.cur.r][node.cur.c].adj[d];
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
    path.score = score;
    const double average = path.score / path.dirs.size();

    Point ite = path.start;
    for (const int& dir : path.dirs)
    {
        Point next = ite;
        next.r += OFS[dir].r;
        next.c += OFS[dir].c;

        grids[ite.r][ite.c].adj[dir] = average;
        grids[next.r][next.c].adj[(dir + 2) % 4] = average;

        ite = next;
    }
}

int main()
{
    for (int r = 0; r < N; ++r)
    {
        for (int c = 0; c < N; ++c)
        {
            for (auto& v : grids[r][c].adj)
            {
                v = static_cast<double>(MIN_DIST + MAX_DIST) / 2.0;
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
    }

    return 0;
}
