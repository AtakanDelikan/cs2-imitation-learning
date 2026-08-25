#include "OptimizedGeometry.h"
#include "Parser.h"
#include <fstream>
#include <iostream>

bool OptimizedGeometry::CreateOptimizedFile(const std::string& rawFile, const std::string& optimizedFile) {
    Parser parser(rawFile);
    meshes = parser.GetCombinedList();

    std::ofstream out(optimizedFile, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << optimizedFile << std::endl;
        return false;
    }
    size_t numMeshes = meshes.size();
    out.write(reinterpret_cast<const char*>(&numMeshes), sizeof(size_t));
    for (const auto& mesh : meshes) {
        size_t numTris = mesh.size();
        out.write(reinterpret_cast<const char*>(&numTris), sizeof(size_t));
        for (const auto& tri : mesh) {
            out.write(reinterpret_cast<const char*>(&tri.v0), sizeof(Vector3));
            out.write(reinterpret_cast<const char*>(&tri.v1), sizeof(Vector3));
            out.write(reinterpret_cast<const char*>(&tri.v2), sizeof(Vector3));
        }
    }
    out.close();
    return true;
}

bool OptimizedGeometry::LoadFromFile(const std::string& optimizedFile) {
    std::ifstream in(optimizedFile, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open optimized file: " << optimizedFile << std::endl;
        return false;
    }
    meshes.clear();
    size_t numMeshes;
    in.read(reinterpret_cast<char*>(&numMeshes), sizeof(size_t));
    for (size_t i = 0; i < numMeshes; ++i) {
        size_t numTris;
        in.read(reinterpret_cast<char*>(&numTris), sizeof(size_t));
        std::vector<TriangleCombined> mesh;
        mesh.resize(numTris);
        for (size_t j = 0; j < numTris; ++j) {
            in.read(reinterpret_cast<char*>(&mesh[j].v0), sizeof(Vector3));
            in.read(reinterpret_cast<char*>(&mesh[j].v1), sizeof(Vector3));
            in.read(reinterpret_cast<char*>(&mesh[j].v2), sizeof(Vector3));
        }
        meshes.push_back(mesh);
    }
    in.close();
    return true;
}

bool OptimizedGeometry::LoadFromTriFile(const std::string& triFile) {
    // 1. Open the file in binary mode
    std::ifstream in(triFile, std::ios::binary | std::ios::ate); // Open at the end to get file size
    if (!in) {
        std::cerr << "Failed to open binary tri file: " << triFile << std::endl;
        return false;
    }

    // 2. Determine how many triangles are in the file based on its byte size
    std::streamsize fileSize = in.tellg();
    in.seekg(0, std::ios::beg); // Rewind back to the start

    // Each Vector3 is 3 floats (3 * 4 bytes = 12 bytes).
    // Each triangle has 3 vertices (3 * 12 bytes = 36 bytes).
    const size_t bytesPerTriangle = sizeof(float) * 9;

    if (fileSize % bytesPerTriangle != 0) {
        std::cerr << "Warning: File size does not match expected triangle alignment. "
            << "It might be corrupted or a different format." << std::endl;
    }

    size_t numTris = fileSize / bytesPerTriangle;

    meshes.clear();
    std::vector<TriangleCombined> mesh;
    mesh.resize(numTris);

    // 3. Read the sequential triangles directly into your vector memory block
    // Because TriangleCombined contains exactly three Vector3s sequentially, 
    // we can read the entire mesh file in one single highly optimized disk operation.
    if (numTris > 0) {
        in.read(reinterpret_cast<char*>(mesh.data()), fileSize);

        if (!in) {
            std::cerr << "Error occurred while reading triangle data stream." << std::endl;
            return false;
        }
    }

    // Push our loaded mesh soup into your collection
    meshes.push_back(std::move(mesh));

    in.close();
    std::cout << "Successfully loaded " << numTris << " triangles from binary file." << std::endl;
    return true;
}
