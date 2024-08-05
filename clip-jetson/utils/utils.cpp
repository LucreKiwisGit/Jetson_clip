#include "utils.h"
#include <string>
#include <fstream>
#include <vector>
#include <iostream>

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> readFileByLines(const std::string& filename) {
    std::vector<std::string> lines;
    std::fstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return lines;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        lines.push_back(line);
    }

    file.close();

    return lines;
}

