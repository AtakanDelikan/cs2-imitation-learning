#pragma once
#include <string>
#include <vector>
#include "Math.hpp"

class OptimizedGeometry {
public:
    std::vector<std::vector<TriangleCombined>> meshes;

    bool LoadFromFile(const std::string& optimizedFile);

    bool LoadFromTriFile(const std::string& optimizedFile);

    bool CreateOptimizedFile(const std::string& rawFile, const std::string& optimizedFile);
};
