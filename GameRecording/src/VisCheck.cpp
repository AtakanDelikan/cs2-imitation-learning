#include "VisCheck.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>

const size_t LEAF_THRESHOLD = 4;

VisCheck::VisCheck(const std::string& optimizedGeometryFile) {
    //if (!geometry.LoadFromFile(optimizedGeometryFile)) {
    //    std::cerr << "Failed to load optimized file: " << optimizedGeometryFile << std::endl;
    //}
    if (!geometry.LoadFromTriFile(optimizedGeometryFile)) {
        std::cerr << "Failed to load optimized file: " << optimizedGeometryFile << std::endl;
    }
    for (const auto& mesh : geometry.meshes) {
        bvhNodes.push_back(BuildBVH(mesh));
    }
}

void VisCheck::ExportToSTL(const std::string & optimizedGeometryFile) {
    std::ofstream stlFile("test.stl");
    if (!stlFile.is_open()) {
        std::cerr << "Failed to create STL file: " << std::endl;
        return;
    }

    if (!geometry.LoadFromTriFile(optimizedGeometryFile)) {
        std::cerr << "Failed to load optimized file: " << optimizedGeometryFile << std::endl;
    }
    //for (const auto& mesh : geometry.meshes) {
    //    bvhNodes.push_back(BuildBVH(mesh));
    //}

    std::vector<TriangleCombined> triangles = geometry.meshes[0];

    // Every STL file begins with a solid name header
    stlFile << "solid exported_map\n";

    for (const auto& tri : triangles) {
        // 1. Calculate the facet normal
        Vector3 edge1 = tri.v1 - tri.v0;
        Vector3 edge2 = tri.v2 - tri.v0;
        Vector3 normal = edge1.cross(edge2);

        // Normalize the normal if your vector class supports it
        float len = std::sqrt(normal.dot(normal));
        if (len > 0.0f) {
            normal = { normal.x / len, normal.y / len, normal.z / len };
        }

        // 2. Write the triangle facet according to STL specifications
        stlFile << "  facet normal " << normal.x << " " << normal.y << " " << normal.z << "\n";
        stlFile << "    outer loop\n";
        stlFile << "      vertex " << tri.v0.x << " " << tri.v0.y << " " << tri.v0.z << "\n";
        stlFile << "      vertex " << tri.v1.x << " " << tri.v1.y << " " << tri.v1.z << "\n";
        stlFile << "      vertex " << tri.v2.x << " " << tri.v2.y << " " << tri.v2.z << "\n";
        stlFile << "    endloop\n";
        stlFile << "  endfacet\n";
    }

    stlFile << "endsolid exported_map\n";
    stlFile.close();
    std::cout << "Successfully exported " << triangles.size() << " triangles" << std::endl;
}

std::unique_ptr<BVHNode> VisCheck::BuildBVH(const std::vector<TriangleCombined>& tris) {
    auto node = std::make_unique<BVHNode>();

    if (tris.empty()) return node;
    AABB bounds = tris[0].ComputeAABB();
    for (size_t i = 1; i < tris.size(); ++i) {
        AABB triAABB = tris[i].ComputeAABB();
        bounds.min.x = std::min(bounds.min.x, triAABB.min.x);
        bounds.min.y = std::min(bounds.min.y, triAABB.min.y);
        bounds.min.z = std::min(bounds.min.z, triAABB.min.z);
        bounds.max.x = std::max(bounds.max.x, triAABB.max.x);
        bounds.max.y = std::max(bounds.max.y, triAABB.max.y);
        bounds.max.z = std::max(bounds.max.z, triAABB.max.z);
    }
    node->bounds = bounds;
    if (tris.size() <= LEAF_THRESHOLD) {
        node->triangles = tris;
        return node;
    }
    Vector3 diff = bounds.max - bounds.min;
    int axis = (diff.x > diff.y && diff.x > diff.z) ? 0 : ((diff.y > diff.z) ? 1 : 2);
    std::vector<TriangleCombined> sortedTris = tris;
    std::sort(sortedTris.begin(), sortedTris.end(), [axis](const TriangleCombined& a, const TriangleCombined& b) {
        AABB aabbA = a.ComputeAABB();
        AABB aabbB = b.ComputeAABB();
        float centerA, centerB;
        if (axis == 0) {
            centerA = (aabbA.min.x + aabbA.max.x) / 2.0f;
            centerB = (aabbB.min.x + aabbB.max.x) / 2.0f;
        }
        else if (axis == 1) {
            centerA = (aabbA.min.y + aabbA.max.y) / 2.0f;
            centerB = (aabbB.min.y + aabbB.max.y) / 2.0f;
        }
        else {
            centerA = (aabbA.min.z + aabbA.max.z) / 2.0f;
            centerB = (aabbB.min.z + aabbB.max.z) / 2.0f;
        }
        return centerA < centerB;
        });

    size_t mid = sortedTris.size() / 2;
    std::vector<TriangleCombined> leftTris(sortedTris.begin(), sortedTris.begin() + mid);
    std::vector<TriangleCombined> rightTris(sortedTris.begin() + mid, sortedTris.end());

    node->left = BuildBVH(leftTris);
    node->right = BuildBVH(rightTris);

    return node;
}

bool VisCheck::IntersectBVH(const BVHNode* node, const Vector3& rayOrigin, const Vector3& rayDir, float maxDistance, float& hitDistance) {
    if (!node->bounds.RayIntersects(rayOrigin, rayDir)) {
        return false;
    }

    bool hit = false;
    if (node->IsLeaf()) {
        for (const auto& tri : node->triangles) {
            float t;
            if (RayIntersectsTriangle(rayOrigin, rayDir, tri, t)) {
                if (t < maxDistance && t < hitDistance) {
                    hitDistance = t;
                    hit = true;
                }
            }
        }
    }
    else {
        if (node->left) {
            hit |= IntersectBVH(node->left.get(), rayOrigin, rayDir, maxDistance, hitDistance);
        }
        if (node->right) {
            hit |= IntersectBVH(node->right.get(), rayOrigin, rayDir, maxDistance, hitDistance);
        }
    }
    return hit;
}

bool VisCheck::IsPointVisible(const Vector3& point1, const Vector3& point2)
{
    Vector3 rayDir = { point2.x - point1.x, point2.y - point1.y, point2.z - point1.z };
    float distance = std::sqrt(rayDir.dot(rayDir));
    rayDir = { rayDir.x / distance, rayDir.y / distance, rayDir.z / distance };
    float hitDistance = std::numeric_limits<float>::max();
    for (const auto& bvhRoot : bvhNodes) {
        if (IntersectBVH(bvhRoot.get(), point1, rayDir, distance, hitDistance)) {
            if (hitDistance < distance) {
                return false;
            }
        }
    }
    return true;
}

bool VisCheck::RayIntersectsTriangle(const Vector3& rayOrigin, const Vector3& rayDir, const TriangleCombined& triangle, float& t)
{
    const float EPSILON = 1e-7f;

    Vector3 edge1 = triangle.v1 - triangle.v0;
    Vector3 edge2 = triangle.v2 - triangle.v0;
    Vector3 h = rayDir.cross(edge2);
    float a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON)
        return false;

    float f = 1.0f / a;
    Vector3 s = rayOrigin - triangle.v0;
    float u = f * s.dot(h);

    if (u < 0.0f || u > 1.0f)
        return false;

    Vector3 q = s.cross(edge1);
    float v = f * rayDir.dot(q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * edge2.dot(q);

    return (t > EPSILON);
}
