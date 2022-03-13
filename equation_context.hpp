#ifndef EQUATION_CONTEXT_HPP_INCLUDED
#define EQUATION_CONTEXT_HPP_INCLUDED

#include "dual_value.hpp"
#include <vector>
#include <utility>
#include <string>
#include <vec/vec.hpp>

struct equation_context
{
    std::vector<std::pair<std::string, dual_types::value>> values;
    std::vector<std::pair<std::string, dual_types::value>> temporaries;

    void pin(dual_types::value& v)
    {
        if(v.is_constant())
            return;

        for(auto& i : temporaries)
        {
            if(dual_types::equivalent(v, i.second))
            {
                dual_types::value facade;
                facade.make_value(i.first);

                v = facade;
                return;
            }
        }

        std::string name = "pv" + std::to_string(temporaries.size());

        dual_types::value old = v;

        temporaries.push_back({name, old});

        dual_types::value facade;
        facade.make_value(name);

        v = facade;
    }

    template<typename T, int N>
    void pin(vec<N, T>& mT)
    {
        for(int i=0; i < N; i++)
        {
            pin(mT[i]);
        }
    }

    void add(const std::string& name, const value& v)
    {
        values.push_back({name, v});
    }

    void build_impl(std::string& argument_string, const std::string& str, const std::map<std::string, std::string>& substitution_map) const
    {
        for(auto& i : values)
        {
            std::string str = "-D" + i.first + "=" + type_to_string(i.second) + " ";

            argument_string += str;
        }

        if(temporaries.size() == 0)
        {
            argument_string += "-DTEMPORARIES" + str + "=DUMMY ";
            return;
        }

        std::string temporary_string;

        for(auto& [current_name, val] : temporaries)
        {
            if(substitution_map.size() != 0)
            {
                dual_types::value cp = val;

                cp.substitute(substitution_map);

                temporary_string += current_name + "=" + type_to_string(cp) + ",";
            }
            else
            {
                temporary_string += current_name + "=" + type_to_string(val) + ",";
            }
        }

        ///remove trailing comma
        if(temporary_string.size() > 0)
            temporary_string.pop_back();

        argument_string += "-DTEMPORARIES" + str + "=" + temporary_string + " ";
    }

    void build(std::string& argument_string, const std::string& str, const std::map<std::string, std::string>& substitution_map) const
    {
        int old_length = argument_string.size();

        build_impl(argument_string, str, substitution_map);

        int new_length = argument_string.size();

        std::cout << "EXTRA LENGTH " << (new_length - old_length) << " " << str << std::endl;
    }

    void build(std::string& argument_string, int idx, const std::map<std::string, std::string>& substitution_map) const
    {
        build(argument_string, std::to_string(idx), substitution_map);
    }
};


#endif // EQUATION_CONTEXT_HPP_INCLUDED
