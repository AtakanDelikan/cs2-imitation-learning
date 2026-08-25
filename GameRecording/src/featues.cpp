#define NOMINMAX
#include <Windows.h>
#include <atomic>
#include <future>
#include <thread>
#include <iostream>
//#include <glm/glm.hpp>
#include "memory.hpp"
#include "client_dll.hpp"
#include "server_dll.hpp"
#include "offsets.hpp"
#include "buttons.hpp"
//#include <sodium.h>
#include <vector>
#include "draw.hpp"
#include <time.h>
#include <chrono>
#include <fstream>
#include "VisCheck.h"
#include <random>
#include <numbers>
#include <memory>
#include <bitset>

#pragma comment (lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")

struct Vector3d {
    float x, y, z;

    Vector3d operator+(const Vector3d& v) const {
        return { x + v.x, y + v.y, z + v.z };
    }

    Vector3d operator-(const Vector3d& v) const {
        return { x - v.x, y - v.y, z - v.z };
    }

    Vector3d operator*(float s) const {
        return { x * s, y * s, z * s };
    }

    Vector3d operator/(float f) const {
        return { x / f, y / f, z / f };
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    
    }
    Vector3d normalized() const {
        float l = length(); return l > 0 ? *this / l : Vector3d{ 0,0,0 };
    }
};

struct Vector2 {
    float x, y;
};

struct Matrix4x4 {
    float m[16];
};

struct QAngle {
    float pitch;
    float yaw;
    float roll;
};

struct PlayerName {
    char name[128];
};

struct Quaternion {
    float x;
    float y;
    float z;
    float w;
};

struct CBoneData {
    Vector3d pos;
    float scale;
    Quaternion rot;
};

struct CBones {
    CBoneData bones[128];
};

struct Bones {
    CBoneData bones[23];
};

struct ShortName {
    char name[10];

    bool operator==(const ShortName& v) const {
        for (int i = 0; i < 10; ++i) {
            if (name[i] != v.name[i])
                return false;
		}
        return true;
    }
};

struct PlayerState {
    Bones bones;
	Vector3d eyePosition;
	Vector3d originPosition;
	Vector3d absVelocity;
	QAngle viewAngles;
    uint64_t buttonMask;
	uint32_t health;
	uint32_t ammoInClip;
	uint16_t weaponID;
    bool isReloading;
	uint64_t steamId;
    ShortName name;
	bool isAlive;
	bool isImmuneToDamage;
	float duckAmount;
	bool onGround;
	float fallVelocity;
	bool isWalking;
	uint8_t teamNum;
    uint32_t tick;
};

struct Tick {
    PlayerState playerState[10];
    uint32_t tick;
	int numPlayers;
};

struct Hitbox_t {
    const char* m_name;           // 0x00 - Pointer to hitbox string name (e.g., "head")
    const char* m_surfaceProperty;// 0x08 - Surface material type
    const char* m_boneName;       // 0x10 - Associated bone name string
    Vector3d m_vMinBounds;         // 0x18 - Local minimum bounding coordinate
    Vector3d m_vMaxBounds;         // 0x24 - Local maximum bounding coordinate
    float m_flShapeRadius;        // 0x30 - Capsule thickness/radius (0.0f if a hard box)
    int32_t m_nBoneId;            // 0x34 - Index of the bone this hitbox is attached to
	// Remaining bytes are padding/flags up to the size of the struct should be 0x70 bytes
};

struct Hitbox {
    Vector3d minBounds;
    Vector3d maxBounds;
    float radius;
    //uint8_t boneId;
};

struct HitboxArray {
	Hitbox hitboxes[19];
};

Vector3d Cross(const Vector3d& a, const Vector3d& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float dot(const Vector3d& a, const Vector3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3d RotateVectorByQuaternion(const Vector3d& v, const Quaternion& q)
{
    Vector3d qv = { q.x, q.y, q.z };
    Vector3d t = Cross(qv, v) * 2.0f;
    return v + (t * q.w) + Cross(qv, t);
}

// Helper to get a random vector perpendicular to a given direction
Vector3d getPerpendicular(const Vector3d& dir) {
    Vector3d tangent = (std::abs(dir.x) > std::abs(dir.z)) ? Vector3d{ -dir.y, dir.x, 0 } : Vector3d{ 0, -dir.z, dir.y };
    return tangent.normalized();
}

void draw_thread() {
    draw::init();
    draw::render();
}

static void sleep_for(double dt)
{
    static constexpr std::chrono::duration<double> MinSleepDuration(0);
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < dt) {
        std::this_thread::sleep_for(MinSleepDuration);
    }
}

bool WorldToScreen(const Vector3d& pos, Vector2& screen, float m[16], int width, int height)
{
    float clipX = pos.x * m[0] + pos.y * m[1] + pos.z * m[2] + m[3];
    float clipY = pos.x * m[4] + pos.y * m[5] + pos.z * m[6] + m[7];
    float clipW = pos.x * m[12] + pos.y * m[13] + pos.z * m[14] + m[15];

    if (clipW < 0.1f)
        return false;

    float ndcX = clipX / clipW;
    float ndcY = clipY / clipW;

    if (ndcX < -1.0f || ndcX > 1.0f || ndcY < -1.0f || ndcY > 1.0f)
        return false;

    screen.x = (width / 2.0f) * (ndcX + 1.0f);
    screen.y = (height / 2.0f) * (1.0f - ndcY);

    //std::cout << "on the point: " << screen.x << "  " << screen.y << std::endl;

    return true;
}

HitboxArray getHitboxArray(const Bones& bones) {
    HitboxArray hitboxes;

    // Helper macro/lambda can clean this up, but keeping it direct for clarity:
    auto transformHitbox = [&](int index, int boneIdx, const Vector3d& localMin, const Vector3d& localMax, float radius) {
        const auto& bone = bones.bones[boneIdx];
        hitboxes.hitboxes[index] = {
            bone.pos + RotateVectorByQuaternion(localMin, bone.rot),
            bone.pos + RotateVectorByQuaternion(localMax, bone.rot),
            radius
        };
        };

    // 0 : 7 head
    transformHitbox(0, 7, { -1, 1.8, 0 }, { 3.5, 0.2, 0 }, 4.3f);
    // 1 : 6 neck
    transformHitbox(1, 6, { 0, -0.4, 0 }, { 1.4, -0.2, 0 }, 3.5f);
    // 2 : 1 hip
    transformHitbox(2, 1, { -2.7, 1.1, -3.2 }, { -2.7, 1.1, 3.2 }, 6.0f);
    // 3 : 2 spine
    transformHitbox(3, 2, { 1.4, 0.8, 3.1 }, { 1.4, 0.8, -3.1 }, 6.0f);
    // 4 : 3 spine
    transformHitbox(4, 3, { 3.8, 0.8, -2.4 }, { 3.8, 0.4, 2.4 }, 6.5f);
    // 5 : 4 spine
    transformHitbox(5, 4, { 4.8, 0.15, -4.1 }, { 4.8, 0.15, 4.1 }, 6.2f);
    // 6 : 5 spine
    transformHitbox(6, 5, { 2.5, -0.6, -6 }, { 2.5, -0.6, 6 }, 5.0f);
    // 7 : 17 upper leg L
    transformHitbox(7, 17, { 1.3, -0.2, 0 }, { 16.5, -0.7, 0 }, 5.0f);
    // 8 : 20 upper leg R
    transformHitbox(8, 20, { -1.3, 0, -0.6 }, { -16.5, 0, -0.7 }, 5.0f);
    // 9 : 18 lower leg L
    transformHitbox(9, 18, { 0.1, -0.4, 0.2 }, { 17, -0.4, 0.7 }, 4.0f);
    // 10 : 21 lower leg R
    transformHitbox(10, 21, { -0.1, 0, -0.2 }, { -17, 0.4, -0.7 }, 4.0f);
    // 11 : 19 foot L
    transformHitbox(11, 19, { -0, -3.43, -0.52 }, { 8, 0.74, 0.33 }, 2.6f);
    // 12 : 22 foot R
    transformHitbox(12, 22, { -7.98, -0.75, -0.27 }, { -0.02, 3.44, 0.58 }, 2.6f);
    // 13 : 11 hand L
    transformHitbox(13, 11, { 0, 0.3, 0 }, { 3.59, 1.15, 0.11 }, 2.3f);
    // 14 : 15 hand R
    transformHitbox(14, 15, { 0, -0.3, 0.02 }, { -3.44, -1.17, -0.09 }, 2.3f);
    // 15 : 9 lower arm L
    transformHitbox(15, 9, { 0, 0, 0 }, { 11.2, 0, 0 }, 3.3f);
    // 16 : 10 upper arm L
    transformHitbox(16, 10, { 0, 0, 0 }, { 10, 0, 0 }, 3.0f);
    // 17 : 13 lower arm R
    transformHitbox(17, 13, { 0, 0, 0 }, { -11.2, 0, 0 }, 3.3f);
    // 18 : 14 upper arm R
    transformHitbox(18, 14, { 0, 0, 0 }, { -10, 0, -0.5 }, 3.0f);

    return hitboxes;
}

void getCapsulePeripheralPoints(Hitbox hitbox, Vector3d eyePos, std::vector<Vector3d>& outPoints, int arcPoints = 7, int linePoints = 4) {
    outPoints.clear();
    if (arcPoints < 2) arcPoints = 2;
    if (linePoints < 2) linePoints = 2;

    // Total points will be exactly (arcPoints * 2) + (linePoints * 2) without duplicates
    outPoints.reserve((arcPoints * 2) + (linePoints * 2));

    Vector3d p1 = hitbox.minBounds; // Bottom sphere center
    Vector3d p2 = hitbox.maxBounds; // Top sphere center
    double r = hitbox.radius;

    Vector3d capDir = p2 - p1;
    double axisLen = capDir.length();
    if (axisLen < 0.0001) return;
    capDir = capDir.normalized();

    // Perspective Horizon Frame for both Spheres
    auto getHorizonFrame = [&](const Vector3d& sphereCenter, Vector3d& outCenter, double& outRadius, Vector3d& outForward) {
        Vector3d toEye = eyePos - sphereCenter;
        double d = toEye.length();
        outForward = toEye.normalized();
        if (d > r) {
            double h = (r * r) / d;
            outRadius = sqrt(r * r - h * h);
            outCenter = sphereCenter + outForward * h;
        }
        else {
            outRadius = r;
            outCenter = sphereCenter;
        }
    };

    Vector3d c2, c1, f2, f1;
    double r2, r1;
    getHorizonFrame(p2, c2, r2, f2);
    getHorizonFrame(p1, c1, r1, f1);

    // Calculate a uniform 3D Side Direction
    Vector3d midCenter = (c1 + c2) * 0.5;
    Vector3d toEyeMid = (eyePos - midCenter).normalized();
    Vector3d sideDir = Cross(capDir, toEyeMid).normalized();

    // Singularity protection (Looking straight down the capsule axis)
    double midDot = (toEyeMid.x * capDir.x) + (toEyeMid.y * capDir.y) + (toEyeMid.z * capDir.z);
    if (midDot > 0.999 || midDot < -0.999) {
        sideDir = (abs(capDir.y) > 0.99) ? Vector3d(1, 0, 0) : Cross(Vector3d(0, 1, 0), capDir).normalized();
    }

    // Local "Up" vectors for the circles that point AWAY from the capsule body
    Vector3d up2 = Cross(f2, sideDir).normalized();
    double up2Dot = (up2.x * capDir.x) + (up2.y * capDir.y) + (up2.z * capDir.z);
    if (up2Dot < 0) up2 = up2 * -1.0;

    Vector3d up1 = Cross(f1, sideDir).normalized();
    double up1Dot = (up1.x * capDir.x) + (up1.y * capDir.y) + (up1.z * capDir.z);
    if (up1Dot > 0) up1 = up1 * -1.0;

    // --- Top Outer Arc ---
    // Exclude the very last point (Left Tangent), because the Left Line starts there.
    for (int i = 0; i < arcPoints; ++i) {
        double t = i / static_cast<double>(arcPoints - 1);
        double theta = t * std::numbers::pi_v<double>;
        Vector3d offset = (sideDir * cos(theta)) + (up2 * sin(theta));
        outPoints.push_back(c2 + offset * r2);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the arc

    // --- Left Connecting Line ---
    // Exclude the very last point (Bottom Left Tangent), because the Bottom Arc starts there.
    Vector3d topLeftTangent = c2 - sideDir * r2;
    Vector3d bottomLeftTangent = c1 - sideDir * r1;
    for (int i = 0; i < linePoints; ++i) {
        double t = i / static_cast<double>(linePoints - 1);
        outPoints.push_back(topLeftTangent * (1.0 - t) + bottomLeftTangent * t);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the line

    // --- Bottom Outer Arc ---
    // Exclude the very last point (Bottom Right Tangent), because the Right Line starts there.
    for (int i = 0; i < arcPoints; ++i) {
        double t = i / static_cast<double>(arcPoints - 1);
        double theta = t * std::numbers::pi_v<double>;
        Vector3d offset = ((sideDir * -1.0) * cos(theta)) + (up1 * sin(theta));
        outPoints.push_back(c1 + offset * r1);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the arc

    // --- Right Connecting Line ---
    // Exclude the very last point (Top Right Tangent), because it matches the exact first point of the Top Arc!
    Vector3d bottomRightTangent = c1 + sideDir * r1;
    Vector3d topRightTangent = c2 + sideDir * r2;
    for (int i = 0; i < linePoints; ++i) {
        double t = i / static_cast<double>(linePoints - 1);
        outPoints.push_back(bottomRightTangent * (1.0 - t) + topRightTangent * t);
    }
    outPoints.pop_back(); // Remove duplicate at the end of the line (loop closes perfectly)
}

Vector2 calculateAngle(const Vector3d& source, const QAngle& viewAngles, const Vector3d& destination) {
    Vector3d delta = destination - source;
    float hypotenuse = std::sqrt(delta.x * delta.x + delta.y * delta.y);

    float targetPitch = std::atan2(-delta.z, hypotenuse) * (180.0f / 3.14159265f);
    float targetYaw = std::atan2(delta.y, delta.x) * (180.0f / 3.14159265f);

    Vector2 relativeAngles;
    relativeAngles.x = targetPitch - viewAngles.pitch;
    relativeAngles.y = targetYaw - viewAngles.yaw;

    while (relativeAngles.y > 180.0f)  relativeAngles.y -= 360.0f;
    while (relativeAngles.y < -180.0f) relativeAngles.y += 360.0f;

    if (relativeAngles.x > 160.0f)  relativeAngles.x = 160.0f;
    if (relativeAngles.x < -160.0f) relativeAngles.x = -160.0f;

    return relativeAngles;
}

bool get_target_point(Vector3d& target, const Bones& bones, const Vector3d& eyePosition, VisCheck& checker) {
    target = { 0 };
    int visiblePoints = 0;
	HitboxArray hitboxes = getHitboxArray(bones);
    int priorityArray[19] = { 0, 1, 6, 5, 4, 3, 2, 15, 17, 16, 18, 13, 14, 7, 8, 9, 10, 11, 12 };

    for (int i = 0; i < 19; ++i) {
        Hitbox hitbox = hitboxes.hitboxes[priorityArray[i]];
        std::vector<Vector3d> points;
        getCapsulePeripheralPoints(hitbox, eyePosition, points, 7, 4);
        for (const Vector3d& point : points) {
            if (checker.IsPointVisible({ eyePosition.x, eyePosition.y, eyePosition.z }, { point.x, point.y, point.z })) {
                visiblePoints++;
                target = target + point;
            }
        }
        if (visiblePoints > 0) {
            target = target / static_cast<float>(visiblePoints);
            return true;
        }
	}

    return false;
}

Matrix4x4 calculateViewMatrix(const Vector3d& camPos, const float& pitch, const float& yaw) {

    float pitchRadians = pitch * (std::numbers::pi_v<float> / 180.f);
    float yawRadians = yaw * (std::numbers::pi_v<float> / 180.f);
    // Pre-calculate the trig once
    float cosPitch = std::cos(pitchRadians);
    float sinPitch = std::sin(pitchRadians);
    float cosYaw = std::cos(yawRadians);
    float sinYaw = std::sin(yawRadians);

    // Derive the 3 orthogonal camera axes
    Vector3d F = { cosPitch * sinYaw,  sinPitch, -cosPitch * cosYaw };
    Vector3d R = { cosYaw,             0.0f,      sinYaw };
    Vector3d U = { -sinYaw * sinPitch, cosPitch,  cosYaw * sinPitch };

    // Calculate the translation offsets via Dot Products
    float tx = -(R.x * camPos.x + R.y * camPos.y + R.z * camPos.z);
    float ty = -(U.x * camPos.x + U.y * camPos.y + U.z * camPos.z);

    // Note: Because the camera looks down -Z, its local Z-axis is (-F). 
    // Therefore the offset is -(-F . Pos) which resolves to +(F . Pos)
    float tz = (F.x * camPos.x + F.y * camPos.y + F.z * camPos.z);

    Matrix4x4 view;

    // Stored in standard Column-Major order:
    // [ m[0]  m[4]  m[8]   m[12] ]
    // [ m[1]  m[5]  m[9]   m[13] ]
    // [ m[2]  m[6]  m[10]  m[14] ]
    // [ m[3]  m[7]  m[11]  m[15] ]

    // Row 0: Right Vector + X-translation
    view.m[0] = R.x;   view.m[4] = R.y;   view.m[8] = R.z;   view.m[12] = tx;

    // Row 1: Up Vector + Y-translation
    view.m[1] = U.x;   view.m[5] = U.y;   view.m[9] = U.z;   view.m[13] = ty;

    // Row 2: Backwards Vector (-Forward) + Z-translation
    view.m[2] = -F.x;  view.m[6] = -F.y;  view.m[10] = -F.z;  view.m[14] = tz;

    // Row 3: Homogeneous row
    view.m[3] = 0.0f;  view.m[7] = 0.0f;  view.m[11] = 0.0f;  view.m[15] = 1.0f;

    return view;
}

Matrix4x4 CalculateUnrealViewMatrix(const Vector3d& camPos, float pitchDeg, float yawDeg) {
    // Convert degrees to Radians
    float pitch = pitchDeg * (std::numbers::pi_v<float> / 180.f);
    float yaw = yawDeg * (std::numbers::pi_v<float> / 180.f);

    float CP = std::cos(pitch);
    float SP = std::sin(pitch);
    float CY = std::cos(yaw);
    float SY = std::sin(yaw);

    // Unreal Engine Left-Handed, Z-Up Basis Vectors:
    // X = Forward into screen, Y = Right, Z = Up
    Vector3d F = { CP * CY,   CP * SY,   SP };
    Vector3d R = { -SY,        CY,        0.0f };
    Vector3d U = { -SP * CY,  -SP * SY,   CP };

    // Translation offsets (Dot products of position against camera axes)
    float tx = -(R.x * camPos.x + R.y * camPos.y + R.z * camPos.z);
    float ty = -(U.x * camPos.x + U.y * camPos.y + U.z * camPos.z);
    float tz = -(F.x * camPos.x + F.y * camPos.y + F.z * camPos.z);

    Matrix4x4 view;

    // Stored Row-by-Row in memory:
    // Row 0: Right Axis
    view.m[0] = R.x;   view.m[1] = R.y;   view.m[2] = R.z;   view.m[3] = tx;
    // Row 1: Up Axis
    view.m[4] = U.x;   view.m[5] = U.y;   view.m[6] = U.z;   view.m[7] = ty;
    // Row 2: Forward Axis
    view.m[8] = F.x;   view.m[9] = F.y;   view.m[10] = F.z;   view.m[11] = tz;
    // Row 3: Homogeneous
    view.m[12] = 0.0f;  view.m[13] = 0.0f;  view.m[14] = 0.0f;  view.m[15] = 1.0f;

    return view;
}

Matrix4x4 CalculateUnrealViewProjection(
    const Vector3d& camPos,
    float pitchDeg, float yawDeg,
    float fovHorizontalDeg = 106.260205f, // Extracted from memory dump
    float screenWidth = 1920.0f,
    float screenHeight = 1080.0f
) {
    // Get the base camera axes (same as above)
    float pitch = -pitchDeg * (3.14159265359f / 180.0f);
    float yaw = yawDeg * (3.14159265359f / 180.0f);

    float CP = std::cos(pitch); float SP = std::sin(pitch);
    float CY = std::cos(yaw);   float SY = std::sin(yaw);

    
    // calculated Right vector is opposite, we negate the Yaw component
    Vector3d F = { CP * CY,   CP * SY,   SP };
    Vector3d R = { SY,       -CY,        0.0f }; // Negated components
    Vector3d U = { -SP * CY,  -SP * SY,   CP };

    // Apply the translation
    float tx = -(R.x * camPos.x + R.y * camPos.y + R.z * camPos.z);
    float ty = -(U.x * camPos.x + U.y * camPos.y + U.z * camPos.z);
    float tz = -(F.x * camPos.x + F.y * camPos.y + F.z * camPos.z);

    // Calculate the Lens zoom scalars
    float halfFovRad = (fovHorizontalDeg * 0.5f) * (3.14159265359f / 180.0f);
    float gx = 1.0f / std::tan(halfFovRad);
    float aspect = screenWidth / screenHeight;
    float gy = gx * aspect; // In memory dump, this resolves strictly to 1.33333f (4/3)

    // Standard gaming Infinite Far-Plane depth scaling
    float nearPlane = 6.5f;
    float farPlane = 102522.0f; // 100000.0f  1.0002522f
    float Q = farPlane / (farPlane - nearPlane);

    Matrix4x4 vp;

    // Row 0: Right axis scaled by Horizontal Zoom
    vp.m[0] = R.x * gx;  vp.m[1] = R.y * gx;  vp.m[2] = R.z * gx;  vp.m[3] = tx * gx;

    // Row 1: Up axis scaled by Vertical Zoom
    vp.m[4] = U.x * gy;  vp.m[5] = U.y * gy;  vp.m[6] = U.z * gy;  vp.m[7] = ty * gy;

    // Row 2: Z-Buffer mapping row (Forward axis * Q)
    vp.m[8] = F.x * Q;   vp.m[9] = F.y * Q;   vp.m[10] = F.z * Q;   vp.m[11] = (tz * Q) - (nearPlane * Q);

    // Row 3: The "Perspective Divide" trigger row (Pure Forward axis)
    vp.m[12] = F.x;       vp.m[13] = F.y;       vp.m[14] = F.z;       vp.m[15] = tz;

    return vp;
}

bool get_target_point_front(Vector3d& target, const Bones& bones, const Vector3d& eyePosition, const QAngle& viewAngles, VisCheck& checker) {
    target = { 0 };
    int visiblePoints = 0;
    HitboxArray hitboxes = getHitboxArray(bones);
    int priorityArray[19] = { 0, 1, 6, 5, 4, 3, 2, 15, 17, 16, 18, 13, 14, 7, 8, 9, 10, 11, 12 };

    Matrix4x4 viewMatrix = CalculateUnrealViewProjection(eyePosition, viewAngles.pitch, viewAngles.yaw);

    for (int i = 0; i < 19; ++i) {
        Hitbox hitbox = hitboxes.hitboxes[priorityArray[i]];
        std::vector<Vector3d> points;
        getCapsulePeripheralPoints(hitbox, eyePosition, points, 7, 4);
        for (const Vector3d& point : points) {
            Vector2 screen;
            if (WorldToScreen(point, screen, viewMatrix.m, 1920, 1080)) {
                if (checker.IsPointVisible({ eyePosition.x, eyePosition.y, eyePosition.z }, { point.x, point.y, point.z })) {
                    visiblePoints++;
                    target = target + point;
                }
            }
        }
        if (visiblePoints > 0) {
            target = target / static_cast<float>(visiblePoints);
            return true;
        }
    }

    return false;
}

void getRelativeVelocity(Vector3d absVelocity, float yaw) {
    float yawRad = yaw * (3.14159265358979323846f / 180.0f);

    float cosYaw = std::cos(yawRad);
    float sinYaw = std::sin(yawRad);

    // Apply 2D rotation matrix by -yaw:
    // localX = absX * cos(yaw) + absY * sin(yaw)   (Forward velocity)
    // localY = -absX * sin(yaw) + absY * cos(yaw)  (Strafe/Side velocity)
    Vector3d relativeVelocity;
    relativeVelocity.x = absVelocity.x * cosYaw + absVelocity.y * sinYaw;
    relativeVelocity.y = -absVelocity.x * sinYaw + absVelocity.y * cosYaw;
    relativeVelocity.z = absVelocity.z; // Vertical velocity stays the same

    std::cout << "yaw " << yaw << std::endl;

    std::cout << "relativeVelocity " << relativeVelocity.x << "|" << relativeVelocity.y << std::endl;
}

void serverScan(HANDLE driver, std::uintptr_t client, std::uintptr_t server) {
    uintptr_t globalVarsPtr = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwGlobalVars);

    //std::system("cls");
    //for (int i = 0; i < 20; i++) {
    //    uint32_t tickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x4*i);
    //    std::cout << "tickBase with offset " << i << " is " << tickBase << std::endl;
    //}

    uint32_t tickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x44);
    //std::cout << "tickBase " << tickBase << std::endl;

    uintptr_t entityList = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwEntityList);

    
    for (int i = 0; i < 64; i++)
    {
        uintptr_t listEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * (i & 0x7FFF) >> 9) + 0x10);

        if (!listEntry)
            continue;

        uintptr_t controller = driver::read_memory<uintptr_t>(driver, listEntry + 0x70 * (i & 0x1FF));

        if (!controller)
            continue;

        uint32_t pawnHandle = driver::read_memory<uint32_t>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_hPawn);
        uintptr_t pawnListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((i & 0x7FFF) >> 9)) + 0x10);
        uintptr_t pawn = driver::read_memory<uintptr_t>(driver, pawnListEntry + 0x70 * (pawnHandle & 0x1FF));

        ShortName name = driver::read_memory<ShortName>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_iszPlayerName);

        if (name.name[0] == '\0')
            continue;

        if (name.name[0] != 'd')
            continue;
        //std::cout << "server name " << name.name << std::endl;

        uintptr_t movementServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pMovementServices);
        Vector3d vecAbsVelocity = driver::read_memory<Vector3d>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_vecAbsVelocity);
        
        //uint64_t buttonMask = driver::read_memory<uint64_t>(driver, movementServices + cs2_dumper::schemas::server_dll::CCSPlayer_MovementServices::m_nButtonDownMaskPrev);
        //uintptr_t movementServices = driver::read_memory<uintptr_t>(driver, playerPawn + cs2_dumper::schemas::server_dll::C_BasePlayerPawn::m_pMovementServices);
        uint64_t buttonStates = driver::read_memory<uint64_t>(driver, movementServices + cs2_dumper::schemas::server_dll::CPlayer_MovementServices::m_nButtons + 0x8 * 1);

        uintptr_t weaponServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pWeaponServices);
        uint32_t weaponHandle = driver::read_memory<uint32_t>(driver, weaponServices + cs2_dumper::schemas::server_dll::CPlayer_WeaponServices::m_hActiveWeapon);
        uintptr_t weaponListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((weaponHandle & 0x7FFF) >> 9)) + 0x10);
        if (!weaponListEntry)
            continue;

        uintptr_t weapon = driver::read_memory<uintptr_t>(driver, weaponListEntry + 0x70 * (weaponHandle & 0x1FF));
        if (!weapon)
            continue;

        uint16_t weaponID = driver::read_memory<uint16_t>(driver, weapon + cs2_dumper::schemas::server_dll::CEconEntity::m_AttributeManager + cs2_dumper::schemas::server_dll::CAttributeContainer::m_Item + cs2_dumper::schemas::server_dll::CEconItemView::m_iItemDefinitionIndex);
        uint32_t ammoInClip = driver::read_memory<uint32_t>(driver, weapon + cs2_dumper::schemas::server_dll::CBasePlayerWeapon::m_iClip1);

        std::cout << tickBase <<  " | ammo " << ammoInClip << " | vel" << vecAbsVelocity.x << vecAbsVelocity.y << " | buttons " << buttonStates << std::endl;
    }
}

float normalizeAngle(float deltaAngle) {
    float maxAngle = 35;
    float power = 0.5; // e.g., 0.5 for square-root scaling

    float clamped = std::clamp(deltaAngle, -maxAngle, maxAngle);
    float sign = (clamped >= 0.0f) ? 1.0f : -1.0f;

    // Ratio in [0.0, 1.0]
    float ratio = std::abs(clamped) / maxAngle;

    // Apply power curve and re-apply sign
    return sign * std::pow(ratio, power);

    return 0.f;
}

void handle_recording(HANDLE driver, std::uintptr_t client, std::uintptr_t server) {
    //std::thread t(draw_thread);

    bool recordingEnabled = false;
    bool messagePrinted = false;
    uint32_t tickBase = 0;

    VisCheck checker = VisCheck::VisCheck("de_dust2.tri");

	//Tick currentTick = {0};
	//Tick previousTick = {0};


    std::ofstream file("server_states.bin", std::ios::binary);
    while (true) {
        //clock_t tStart = clock();
        //auto start = std::chrono::high_resolution_clock::now();
        if (GetAsyncKeyState(VK_END))
        {
            // Force the window to close so draw::render() returns
            //PostMessage(FindWindowA("OverlayWindowClass", "Overlay Window"), WM_CLOSE, 0, 0); // uncomment for drawing boxes
            file.close();
            break;
        }
            
        const int recordingKey = 0x71; // F2 key
        if (GetAsyncKeyState(recordingKey) & 1) {
            recordingEnabled = !recordingEnabled;
            messagePrinted = false;
        }
        if (recordingEnabled && !messagePrinted) {
            std::cout << "Recording is ON" << std::endl;
            messagePrinted = true;
        }
        else if (!recordingEnabled && !messagePrinted) {
            std::cout << "Recording is OFF" << std::endl;
            messagePrinted = true;
        }
        if (recordingEnabled) {
            //draw::clear();
            //draw::add_box(0, 0);

            //serverScan(driver, client, server);
            //continue;

            uintptr_t entityList = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwEntityList);
            
            if (tickBase == 0) {
                uintptr_t globalVarsPtr = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwGlobalVars);
                tickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x44);
                if (tickBase == 0)
                    continue;
			}
            else {
                uintptr_t globalVarsPtr = driver::read_memory<uintptr_t>(driver, server + cs2_dumper::offsets::server_dll::dwGlobalVars);
                uint32_t currentTickBase = driver::read_memory<uint32_t>(driver, globalVarsPtr + 0x44);
                if (currentTickBase == tickBase) {
                    continue;
                }
                sleep_for(0.001); // sleep for 1ms to read more stable data
                if (currentTickBase > tickBase + 1) {
                    std::cout << "JUMP HAPPENED!!!!!!!!!!!!!!" << tickBase << std::endl;
                }
				tickBase = currentTickBase;
            }

            //draw::clear();
			int currentPlayerIndex = 0;
            //previousTick = currentTick;
            for (int i = 0; i < 64; i++)
            {
                uintptr_t listEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * (i & 0x7FFF) >> 9) + 0x10);
                
                if (!listEntry)
                    continue;

                uintptr_t controller = driver::read_memory<uintptr_t>(driver, listEntry + 0x70 * (i & 0x1FF));

                if (!controller)
                    continue;

                uint32_t pawnHandle = driver::read_memory<uint32_t>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_hPawn);
                uintptr_t pawnListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((i & 0x7FFF) >> 9)) + 0x10);
                uintptr_t pawn = driver::read_memory<uintptr_t>(driver, pawnListEntry + 0x70 * (pawnHandle & 0x1FF));

                ShortName name = driver::read_memory<ShortName>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_iszPlayerName);

                if (name.name[0] == '\0')
					continue;

                bool isAlive = driver::read_memory<bool>(driver, controller + cs2_dumper::schemas::server_dll::CCSPlayerController::m_bPawnIsAlive);
                if (!isAlive)
                    continue;

                uintptr_t weaponServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pWeaponServices);
                uint32_t weaponHandle = driver::read_memory<uint32_t>(driver, weaponServices + cs2_dumper::schemas::server_dll::CPlayer_WeaponServices::m_hActiveWeapon);
                uintptr_t weaponListEntry = driver::read_memory<uintptr_t>(driver, entityList + (0x8 * ((weaponHandle & 0x7FFF) >> 9)) + 0x10);
                if (!weaponListEntry)
                    continue;

                uintptr_t weapon = driver::read_memory<uintptr_t>(driver, weaponListEntry + 0x70 * (weaponHandle & 0x1FF));
                if (!weapon)
                    continue;

                uint16_t weaponID = driver::read_memory<uint16_t>(driver, weapon + cs2_dumper::schemas::server_dll::CEconEntity::m_AttributeManager + cs2_dumper::schemas::server_dll::CAttributeContainer::m_Item + cs2_dumper::schemas::server_dll::CEconItemView::m_iItemDefinitionIndex);
                uint32_t ammoInClip = driver::read_memory<uint32_t>(driver, weapon + cs2_dumper::schemas::server_dll::CBasePlayerWeapon::m_iClip1);
                bool inReload = driver::read_memory<bool>(driver, weapon + cs2_dumper::schemas::server_dll::CCSWeaponBase::m_bInReload);

                auto flags = driver::read_memory<std::uint32_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_fFlags);
                bool onGround = flags & (1 << 0);

                uintptr_t bodyComponent = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_CBodyComponent);
                uintptr_t gameScene = driver::read_memory<uintptr_t>(driver, bodyComponent + cs2_dumper::schemas::server_dll::CBodyComponent::m_pSceneNode);
                Vector3d origin = driver::read_memory<Vector3d>(driver, gameScene + cs2_dumper::schemas::server_dll::CGameSceneNode::m_vecAbsOrigin);
                Vector3d viewOffset = driver::read_memory<Vector3d>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseModelEntity::m_vecViewOffset);
                Vector3d eyePosition = origin + viewOffset;
                Vector3d vecAbsVelocity = driver::read_memory<Vector3d>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_vecAbsVelocity);
                QAngle viewAngles = driver::read_memory<QAngle>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_angEyeAngles);

                bool isWalking = driver::read_memory<bool>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_bIsWalking);
                uint8_t teamNum = driver::read_memory<uint8_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_iTeamNum);
                uintptr_t CSkeletonInstance = driver::read_memory<uintptr_t>(driver, gameScene + cs2_dumper::schemas::server_dll::CSkeletonInstance::m_modelState + cs2_dumper::schemas::server_dll::CBodyComponentSkeletonInstance::m_skeletonInstance);
                Bones bones = driver::read_memory<Bones>(driver, CSkeletonInstance);
                uintptr_t movementServices = driver::read_memory<uintptr_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBasePlayerPawn::m_pMovementServices);
				float duckAmount = driver::read_memory<float>(driver, movementServices + cs2_dumper::schemas::server_dll::CCSPlayer_MovementServices::m_flDuckAmount);
                uint64_t steamId = driver::read_memory<uint64_t>(driver, controller + cs2_dumper::schemas::server_dll::CBasePlayerController::m_steamID);
				bool isImmune = driver::read_memory<bool>(driver, pawn + cs2_dumper::schemas::server_dll::CCSPlayerPawn::m_bGunGameImmunity);
                uint32_t health = driver::read_memory<uint32_t>(driver, pawn + cs2_dumper::schemas::server_dll::CBaseEntity::m_iHealth);
                float fallVelocity = driver::read_memory<float>(driver, movementServices + cs2_dumper::schemas::server_dll::CPlayer_MovementServices_Humanoid::m_flFallVelocity);
                uint64_t buttonMask = driver::read_memory<uint64_t>(driver, movementServices + cs2_dumper::schemas::server_dll::CPlayer_MovementServices::m_nButtons + 0x8 * 1);


				PlayerState playerState;
                playerState.bones = std::move(bones);
                playerState.eyePosition = std::move(eyePosition);
                playerState.originPosition = std::move(origin);
                playerState.absVelocity = std::move(vecAbsVelocity);
                playerState.viewAngles = std::move(viewAngles);
                playerState.buttonMask = std::move(buttonMask);
                playerState.health = std::move(health);
                playerState.ammoInClip = std::move(ammoInClip);
                playerState.weaponID = std::move(weaponID);
                playerState.isReloading = std::move(inReload);
                playerState.steamId = std::move(steamId);
				playerState.name = std::move(name);
                playerState.isAlive = std::move(isAlive);
                playerState.isImmuneToDamage = std::move(isImmune);
                playerState.duckAmount = std::move(duckAmount);
                playerState.onGround = std::move(onGround);
                playerState.fallVelocity = std::move(fallVelocity);
                playerState.isWalking = std::move(isWalking);
				playerState.teamNum = std::move(teamNum);
                playerState.tick = std::move(tickBase);
               

                file.write( 
                    reinterpret_cast<const char*>(&playerState),
                    sizeof(PlayerState)
                );

            }

        }
    }
}



/*
For each player record:
whole bone matrix
eye position
origin position
absolute velocity
view angles
view angle velocity
health
their ammo counts
what weapon they have
steam id
isAlive
isImmuneToDamage
duck amount
on ground flag
fall velocity
isReloading
is walking flag
team number
*/