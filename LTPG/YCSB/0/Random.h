#pragma once

#include <string>
#include <vector>
#include <ctime>
class Random
{
public:
    Random(uint64_t seed = 0)
    {
        init_seed(seed);
        srand((uint16_t)time(0));
    }

    void init_seed(uint64_t seed)
    {
        seed_ = (seed ^ 0x5DEECE66DULL) & ((1ULL << 48) - 1);
    }

    void set_seed(uint64_t seed)
    {
        seed_ = seed;
    }

    uint64_t get_seed()
    {
        return seed_;
    }

    uint64_t next()
    {
        return ((uint64_t)next(32) << 32) + next(32);
    }

    uint64_t next(unsigned int bits)
    {
        seed_ = (seed_ * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
        return (seed_ >> (48 - bits));
    }

    /* [0.0, 1.0) */
    double next_double()
    {
        return (((uint64_t)next(26) << 27) + next(27)) / (double)(1ULL << 53);
    }

    uint64_t uniform_dist(uint64_t a, uint64_t b)
    {
        if (a == b)
            return a;
        // return next() % (b - a + 1) + a;
        return rand() % (b - a + 1) + a;
    }

    std::string rand_str(std::size_t length, const std::string &str)
    {
        std::string result;
        auto str_len = str.length();
        for (auto i = 0u; i < length; i++)
        {
            int k = uniform_dist(0, str_len - 1);
            result += str[k];
        }
        return result;
    }

    std::string a_string(std::size_t min_len, std::size_t max_len)
    {
        auto len = uniform_dist(min_len, max_len);
        return rand_str(len, alpha());
    }
    uint64_t non_uniform_distribution(uint64_t A, uint64_t x, uint64_t y)
    {
        return (uniform_dist(0, A) | uniform_dist(x, y)) % (y - x + 1) + x;
    }

    std::string n_string(std::size_t min_len, std::size_t max_len)
    {
        auto len = uniform_dist(min_len, max_len);
        return rand_str(len, numeric());
    }

    std::string rand_zip()
    {
        auto zip = n_string(4, 4);
        // append "11111"
        for (int i = 0; i < 5; i++)
        {
            zip += '1';
        }
        return zip;
    }

    std::string rand_last_name(int n)
    {
        const auto &last_names = customer_last_names();
        const auto &s1 = last_names[n / 100];
        const auto &s2 = last_names[n / 10 % 10];
        const auto &s3 = last_names[n % 10];
        return s1 + s2 + s3;
    }

private:
    static const std::string &alpha()
    {
        static std::string alpha_ =
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        return alpha_;
    };
    static const std::vector<std::string> &customer_last_names()
    {
        static std::vector<std::string> last_names = {
            "BAR", "OUGHT", "ABLE", "PRI", "PRES",
            "ESE", "ANTI", "CALLY", "ATION", "EING"};
        return last_names;
    };

    static const std::string &numeric()
    {
        static std::string numeric_ = "0123456789";
        return numeric_;
    };
    uint64_t seed_;
};