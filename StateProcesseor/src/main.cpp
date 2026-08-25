#include <iostream>
#include <fstream>
#include <vector>
#include <numbers>
#include "VisCheck.h"
#include <deque>
#include <thread>

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
    float distance(const Vector3d& other) const {
        return (*this - other).length();
    }
    Vector3d normalized() const {
        float l = length(); return l > 0 ? *this / l : Vector3d{ 0,0,0 };
    }

};

struct Vector2 {
    float x, y;
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

struct Bones {
    CBoneData bones[23];
};

struct Matrix4x4 {
    float m[16];
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

    bool operator!=(const ShortName& v) const {
        return !(name == v.name);
    }
};

struct PlayerState {
    Bones bones;
    Vector3d eyePosition;
    Vector3d originPosition;
    Vector3d absVelocity;
    QAngle viewAngles;
    //QAngle viewAngleVelocity;
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

struct PlayerAction {
    // Continuous: Aiming
    float deltaPitch; // Vertical aim movement
    float deltaYaw; // Horizontal aim movement

    // Discrete: Movement Axes (-1, 0, or 1)
    float moveX;      // S = -1, None or both = 0, W = 1
    float moveY;      // A = -1, None or both = 0, D = 1

    // Binary: Combat and Utility
    bool isShooting;   // Left click
    bool isJumping;    // Space
    bool isCrouching;  // Ctrl
    bool isReloading;  // R
    bool isWalking;    // Shift
};

struct EnemyObservation {
    bool isValid; // stays true for 90 ticks after an enemy is seen, then becomes false until a new enemy is seen
    bool isVisible; // not visible if not rendered on screen
    bool isImmune; // is immune to damage in deathmatch mode
    float proximityScore;
    float targetDeltaPitch;
    float targetDeltaYaw;
    float memoryConfidence; // [0.0f, 1.0f] (1.0 - ticks/90.0)
    ShortName name; // just for calculations, not included in the training data.
    Vector3d lastSeenPos; // just for calculations, not included in the training data.
    int ticksSinceLastSeen; // just for calculations, not included in the training data.
};

struct DamageObservation {
    bool isValid;            // 1.0 if hit recently, 0.0 if empty padding
    float damageDeltaPitch;    // Relative horizontal angle (-1.0 to 1.0)
    float damageDeltaYaw;  // Relative vertical angle (-1.0 to 1.0)
    float damageIntensity;   // timer that starts at 1.0 when hit and decays to 0.0 over 30 ticks
    Vector3d source; //field to prepare the data, to be deleted after preparation
    ShortName name; //field to prepare the data, to be deleted after preparation
};

struct SelfObservation {
    // GLOBAL: Where am I? (Lets the bot memorize the map)
    float globalPosX;
    float globalPosY;
    float globalPosZ;

    // Where am I looking at?
    float pitch;
    float yaw;

    // Movement State
    float velocityForward;  // Relative to crosshair (W/S)
    float velocityRight;    // Relative to crosshair (A/D)
    float velocityUp;       // Falling or jumping speed

    // Stance & Environment Flags
    float isGrounded;
    float duckAmount;       // 0.0 (standing) to 1.0 (fully crouched)

    // Combat State
    float healthPercentage; // 0.0 to 1.0
    float ammoPercentage;   // 0.0 to 1.0
    float isReloading; // 0.0 or 1.0
};

struct FlatFrame {
    float values[65]; // 9+7*5+4*2+13
};

struct flatPlayerAction {
    // Continuous: Aiming
    float actionDeltaPitch; // Vertical aim movement
    float actionDeltaYaw; // Horizontal aim movement

    // Discrete: Movement Axes (-1, 0, or 1)
    float actionMoveX;      // S = -1, None or both = 0, W = 1
    float actionMoveY;      // A = -1, None or both = 0, D = 1

    // Binary: Combat and Utility
    float actionIsShooting;   // Left click
    float actionIsJumping;    // Space
    float actionIsCrouching;  // Ctrl
    float actionIsReloading;  // R
    float actionIsWalking;    // Shift
};

struct flatEnemyObservation {
    float isValid; // stays true for 90 ticks after an enemy is seen, then becomes false until a new enemy is seen
    float isVisible; // not visible if not rendered on screen
    float isImmune; // is immune to damage in deathmatch mode
    float proximityScore;
    float targetDeltaPitch;
    float targetDeltaYaw;
    float memoryConfidence; // [0.0f, 1.0f] (1.0 - ticks/90.0)
};

struct flatDamageObservation {
    float isValid;            // 1.0 if hit recently, 0.0 if empty padding
    float damageDeltaPitch;    // Relative horizontal angle (-1.0 to 1.0)
    float damageDeltaYaw;  // Relative vertical angle (-1.0 to 1.0)
    float damageIntensity;   // timer that starts at 1.0 when hit and decays to 0.0 over 30 ticks
};

struct flatSelfObservation {
    // GLOBAL: Where am I? (Lets the bot memorize the map)
    float globalPosX;
    float globalPosY;
    float globalPosZ;

    // Where am I looking at?
    float pitch;
    float yaw;

    // Movement State
    float velocityForward;  // Relative to crosshair (W/S)
    float velocityRight;    // Relative to crosshair (A/D)
    float velocityUp;       // Falling or jumping speed

    // Stance & Environment Flags
    float isGrounded;
    float duckAmount;       // 0.0 (standing) to 1.0 (fully crouched)

    // Combat State
    float healthPercentage; // 0.0 to 1.0
    float ammoPercentage;   // 0.0 to 1.0
    float isReloading; // 0.0 or 1.0
};

struct SteamIDList {
    uint64_t id[50];
};

struct NameList {  // TODO instead of having a fixed size list, use vector
    ShortName name[50];
};

bool inList(uint64_t id, const SteamIDList& list) {
    for (const auto& steamId : list.id) {
        if (steamId == id) {
            return true;
        }
    }
    return false;
}

bool inNameList(const ShortName& name, const NameList& list) {
    for (const auto& n : list.name) {
        if (n == name) {
            return true;
        }
    }
    return false;
}

bool inNameList(const ShortName& name, const std::vector<ShortName>& playersList) {
    return std::find(playersList.begin(), playersList.end(), name) != playersList.end();
}

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

Vector3d RotateVectorByQuaternion(const Vector3d& v, const Quaternion& q)
{
    Vector3d qv = { q.x, q.y, q.z };
    Vector3d t = Cross(qv, v) * 2.0f;
    return v + (t * q.w) + Cross(qv, t);
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

Vector3d getLocalVelocity(Vector3d absVelocity, float yaw) {
    float yawRad = yaw * (std::numbers::pi_v<float> / 180.0f);

    float cosYaw = std::cos(yawRad);
    float sinYaw = std::sin(yawRad);

    // Apply 2D rotation matrix by -yaw:
    // localX = absX * cos(yaw) + absY * sin(yaw)   (Forward velocity)
    // localY = -absX * sin(yaw) + absY * cos(yaw)  (Strafe/Side velocity)
    Vector3d relativeVelocity;
    relativeVelocity.x = absVelocity.x * cosYaw + absVelocity.y * sinYaw;
    relativeVelocity.y = -absVelocity.x * sinYaw + absVelocity.y * cosYaw;
    relativeVelocity.z = absVelocity.z; // Vertical velocity stays the same

    return relativeVelocity;
}

// Function to linearly normalize a value to the range [-1.0, 1.0]
float normalize(float value, float origMin, float origMax) {
    float normalized = -1.0f + ((value - origMin) * 2.0f) / (origMax - origMin);
    float clamped = std::clamp(normalized, -1.f, 1.f);
    return clamped;
}

// Function to normalize angle. 35 for mouse movement, 60 for the angle between crosshair and enemy
float normalizeAngle(float deltaAngle, float maxAngle = 35) {
    float power = 0.5; // e.g., 0.5 for square-root scaling

    float clamped = std::clamp(deltaAngle, -maxAngle, maxAngle);
    float sign = (clamped >= 0.0f) ? 1.0f : -1.0f;

    // Ratio in [0.0, 1.0]
    float ratio = std::abs(clamped) / maxAngle;

    // Apply power curve and re-apply sign
    return sign * std::pow(ratio, power);

    return 0.f;
}


// Proximity Score[0.0 = Far / Irrelevant, 1.0 = Right on top of player]
// combat_radius: Distance in units where enemy importance drops by ~63% (3000.0f is max distance for dust2)
float getProximityScore(float distance, float combatRadius = 500.0f) {
    return std::exp(-distance / combatRadius);
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

    // 1. Perspective Horizon Frame for both Spheres
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
        sideDir = (abs(capDir.y) > 0.99) ? Vector3d{ 1, 0, 0 } : Cross(Vector3d{ 0, 1, 0 }, capDir).normalized();
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

bool get_target_point(Vector3d& target, const Bones& bones, const Vector3d& eyePosition, const VisCheck& checker) {
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

Matrix4x4 CalculateUnrealViewProjection(
    const Vector3d& camPos,
    float pitchDeg, float yawDeg,
    float fovHorizontalDeg = 106.260205f, // Extracted from memory dump
    float screenWidth = 1920.0f,
    float screenHeight = 1080.0f
) {
    // Get the base camera axes
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

bool get_target_point_front(Vector3d& target, const Bones& bones, const Vector3d& eyePosition, const QAngle& viewAngles, const VisCheck& checker) {
    target = { 0 };
    int visiblePoints = 0;
    HitboxArray hitboxes = getHitboxArray(bones);
    int priorityArray[19] = { 0, 1, 6, 5, 4, 3, 2, 15, 17, 16, 18, 13, 14, 7, 8, 9, 10, 11, 12 };
    //int priorityArray[19] = { 0, 1, 6, 5, 4, 3, 2, 15, 17, 16, 18, 13, 14, 7, 8, 9, 10, 11, 12 };

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

void analyzeDamage(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const NameList& nameList, const VisCheck& checker, std::vector<DamageObservation>& damageObservations, int whichPlayer)
{
    int currentHealth = 0;
    int previousHealth = 0;
    PlayerState targetState;
    bool found = false;
    for (const auto& curState : curStates) {
        if (curState.name == nameList.name[whichPlayer]) {
            currentHealth = curState.health;
            targetState = curState;
            found = true;
            break;
        }
    }
    if (!found) {
        return;
    }
    found = false;
    for (const auto& prevState : prevStates) {
        if (prevState.name == nameList.name[whichPlayer]) {
            previousHealth = prevState.health;
            found = true;
            break;
        }
    }

    if (!found) {
        // the player wasnt in the previous Tick. must be spawned this tick.
        // TODO copy curState to prevState
    }

    int framesToFade = 30;
    float fadeAmount = 1.0f / framesToFade;
    // lower the damage observation intensity by 1/30, if its 0 make other fields 0.
    for (auto& damageObservation : damageObservations) {
        damageObservation.damageIntensity -= fadeAmount;
        if (damageObservation.damageIntensity <= 0) {
            damageObservation = { 0 };
        }
        else {
            Vector2 angle = calculateAngle(targetState.eyePosition, targetState.viewAngles, damageObservation.source);
            damageObservation.damageDeltaPitch = normalizeAngle(angle.x, 180);
            damageObservation.damageDeltaYaw = normalizeAngle(angle.y, 180);
        }
    }

    if (currentHealth < previousHealth) {
        //std::cout << nameList.name[whichPlayer].name << " Got damaged at tick " << curStates[0].tick << " from " << previousHealth << " to " << currentHealth << std::endl;
    }
    else {
        return;
    }

    for (const auto& curState : curStates) {
        if (curState.name == nameList.name[whichPlayer]) {
            continue;
        }
        for (const auto& prevState : prevStates) {
            if (curState.name == prevState.name) {
                int currentAmmo = curState.ammoInClip;
                int previousAmmo = prevState.ammoInClip;
                if (currentAmmo < previousAmmo) {
                    Vector3d target;
                    bool visible = get_target_point_front(target, targetState.bones, curState.eyePosition, curState.viewAngles, checker);
                    if (visible) {
                        //std::cout << " damage " << curState.eyePosition.x << "|" << curState.eyePosition.y << "|" << curState.eyePosition.z << " || "
                        //    << targetState.eyePosition.x << "|" << targetState.eyePosition.y << "|" << targetState.eyePosition.z << std::endl;
                        //std::cout << " angle " << curState.viewAngles.pitch << "|" << curState.viewAngles.yaw << std::endl;
                        
                        
                        //Vector2 angle = calculateAngle(curState.eyePosition, curState.viewAngles, targetState.eyePosition);
                        Vector2 angle = calculateAngle(targetState.eyePosition, targetState.viewAngles, curState.eyePosition);
                        //std::cout << "Player " << nameList.name[whichPlayer].name << " damaged by " << curState.name.name << " at tick " << curState.tick
                        //    << " angle " << angle.x << " | " << angle.y << std::endl;

                        //std::cout << "shooter pos " << curState.eyePosition.x << " | " << curState.eyePosition.y << " looking at "
                        //    << curState.viewAngles.pitch << "|" << curState.viewAngles.yaw << std::endl;
                        //std::cout << "target pos " << targetState.eyePosition.x << " | " << targetState.eyePosition.y << " looking at "
                        //    << targetState.viewAngles.pitch << "|" << targetState.viewAngles.yaw << std::endl;
                        //if the name is in damage observations override it, otherwise override the observation with the least intensity
                        DamageObservation *leastIntenseDO = &damageObservations.front();
                        for (auto& damageObservation : damageObservations) {
                            if (damageObservation.name == curState.name) {
                                damageObservation.isValid = true;
                                damageObservation.damageDeltaPitch = normalizeAngle(angle.x, 180);
                                damageObservation.damageDeltaYaw = normalizeAngle(angle.y, 180);
                                damageObservation.damageIntensity = 1.0;
                                damageObservation.source = curState.eyePosition;
                                damageObservation.name = curState.name;
                                return; // TODO why is this here?
                            }
                            if (damageObservation.damageIntensity < leastIntenseDO->damageIntensity) {
                                leastIntenseDO = &damageObservation;
                            }
                        }
                        leastIntenseDO->isValid = true;
                        leastIntenseDO->damageDeltaPitch = angle.x;
                        leastIntenseDO->damageDeltaYaw = angle.y;
                        leastIntenseDO->damageIntensity = 1.0;
                        leastIntenseDO->source = curState.eyePosition;
                        leastIntenseDO->name = curState.name;
                    }
                }
            }
        }
    }
    
}

void replaceEnemyObservation(std::vector<EnemyObservation>& enemyObservations, const EnemyObservation& newObservation) {

    // Override observation with the same name if it exists
    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.name == newObservation.name) {
            enemyObservation = newObservation;
            return;
        }
    }

    // Override not valid observation if there is any
    for (auto& enemyObservation : enemyObservations) {
        if (!enemyObservation.isValid) {
            enemyObservation = newObservation;
            return;
        }
    }

    // Override oldest not visible if there is any
    EnemyObservation* observationToReplace = nullptr;
    int maxAge = -1;

    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && !enemyObservation.isVisible) {
            int age = enemyObservation.ticksSinceLastSeen;
            if (age > maxAge) {
                maxAge = age;
                observationToReplace = &enemyObservation;
            }
        }
    }
    if (observationToReplace != nullptr) {
        *observationToReplace = newObservation;
        return;
    }
    

    // Override valid, visible with highest delta angle
    observationToReplace = nullptr;
    float maxAngularDistance = -1.f;

    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && enemyObservation.isVisible) {
            float AngularDistance = std::hypot(enemyObservation.targetDeltaPitch, enemyObservation.targetDeltaYaw);
            if (AngularDistance > maxAngularDistance) {
                maxAngularDistance = AngularDistance;
                observationToReplace = &enemyObservation;
            }
        }
    }
    if (observationToReplace != nullptr) {
        *observationToReplace = newObservation;
        return;
    }

    std::cout << "never should have happened :)" << std::endl; // should never happen
}

void reorderEnemyObservations(std::vector<EnemyObservation>& enemyObservations) {
    std::sort(enemyObservations.begin(), enemyObservations.end(), [](const EnemyObservation& a, const EnemyObservation& b) {
        // --- Handle validity grouping ---
        // Valid elements come before invalid elements
        if (a.isValid != b.isValid) {
            return a.isValid > b.isValid; // true (1) comes before false (0)
        }

        // If both are invalid, their relative order doesn't matter
        if (!a.isValid) {
            return false;
        }

        // --- Handle visibility grouping (both are valid here) ---
        // Visible elements come before non-visible elements
        if (a.isVisible != b.isVisible) {
            return a.isVisible > b.isVisible;
        }

        // --- Tie-breakers within the same group ---
        if (a.isVisible) {
            // 1st Priority: Order by least hypot(deltaPitch, deltaYaw)
            // Using squared values avoids slow sqrt() calls
            float distSqA = (a.targetDeltaPitch * a.targetDeltaPitch) + (a.targetDeltaYaw * a.targetDeltaYaw);
            float distSqB = (b.targetDeltaPitch * b.targetDeltaPitch) + (b.targetDeltaYaw * b.targetDeltaYaw);
            return distSqA < distSqB;
        }
        else {
            // 2nd Priority: Order by age / ticksSinceLastSeen (ascending/youngest first)
            return a.ticksSinceLastSeen < b.ticksSinceLastSeen;
        }
        });
}

void analyzeEnemy(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const NameList& nameList, const VisCheck& checker, std::vector<EnemyObservation>& enemyObservations, int whichPlayer)
{
    // TODO should this function return isPlayerAlive? so that we wont produce processed state data for our ml model?

    PlayerState playerState;
    //NameList alivePlayersList;
    std::vector<ShortName> alivePlayersList;
    std::vector<ShortName> seenPlayersList;
    for (const auto& curState : curStates) {
        if (curState.name == nameList.name[whichPlayer]) {
            playerState = curState;
        }
        else {
            if (!inNameList(curState.name, alivePlayersList)) {
                alivePlayersList.push_back(curState.name);
            }
        }
    }

    std::vector<EnemyObservation> newObservations;
    for (const auto& curState : curStates) {
        if (curState.name == nameList.name[whichPlayer]) {
            continue;
        }
        else {
            Vector3d target;
            bool visible = get_target_point_front(target, curState.bones, playerState.eyePosition, playerState.viewAngles, checker);
            if (visible) {
                seenPlayersList.push_back(curState.name);
                Vector2 angle = calculateAngle(playerState.eyePosition, playerState.viewAngles, target);

                float distance = playerState.eyePosition.distance(target);
                float proximity = getProximityScore(distance);
                EnemyObservation newObservation = {
                    .isValid = true,
                    .isVisible = true,
                    .isImmune = curState.isImmuneToDamage,
                    .proximityScore = proximity,
                    .targetDeltaPitch = normalizeAngle(angle.x ,60), // max angle is around 60 in the game with default settings
                    .targetDeltaYaw = normalizeAngle(angle.y ,60), // max angle is around 60 in the game with default settings
                    .memoryConfidence = 1,
                    .name = curState.name,
                    .lastSeenPos = target,
                    .ticksSinceLastSeen = 0
                };

                newObservations.push_back(newObservation);

                //replaceEnemyObservation(enemyObservations, newObservation);

            }
        }
    }

    int ticksToForget = 90;
    for (auto& enemyObservation : enemyObservations) {
        if (enemyObservation.isValid && enemyObservation.isVisible) {
            // if enemy was valid and visible but not visible this tick make it not visible
            if (!inNameList(enemyObservation.name, seenPlayersList)) {
                enemyObservation.isVisible = false;
            }
        }
        if (enemyObservation.isValid && !enemyObservation.isVisible) {
            enemyObservation.ticksSinceLastSeen += 1;
            enemyObservation.memoryConfidence = 1.0 - (enemyObservation.ticksSinceLastSeen / 90.0);
            // update delta angle, proximityScore
            Vector2 angle = calculateAngle(playerState.eyePosition, playerState.viewAngles, enemyObservation.lastSeenPos);
            enemyObservation.targetDeltaPitch = normalizeAngle(angle.x, 60); // max angle is around 60 in the game with default settings
            enemyObservation.targetDeltaYaw = normalizeAngle(angle.y, 60); // max angle is around 60 in the game with default settings
            float distance = playerState.eyePosition.distance(enemyObservation.lastSeenPos);
            enemyObservation.proximityScore = getProximityScore(distance);
        }
        if (enemyObservation.ticksSinceLastSeen > ticksToForget) {
            enemyObservation = { 0 };
        }
        if (!inNameList(enemyObservation.name, alivePlayersList)) { // forget enemy if dies this tick
            enemyObservation = { 0 };
        }
        if (!enemyObservation.isValid) {
            enemyObservation = { 0 };
        }
        //enemyObservation.isImmune = ; // TODO figure out how to update this, now only updates when enemy is visible, which might actually be ok
    }

    // If new observations are enough to fill enemyObservations fill them
    if (newObservations.size() >= enemyObservations.size()) {
        reorderEnemyObservations(newObservations);

        // Store the target size before changing anything
        size_t targetSize = enemyObservations.size();

        // Safely replace the contents with the range from newObservations
        enemyObservations.assign(newObservations.begin(), newObservations.begin() + targetSize);
        return;
    }

    // there is less newObservations than enemyObservations slots
    for (auto& newObservation : newObservations) {
        replaceEnemyObservation(enemyObservations, newObservation);
    }

    reorderEnemyObservations(enemyObservations);
}

void analyzeSelf(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const NameList& nameList, SelfObservation& selfObs, int whichPlayer) {
    bool found = false;
    PlayerState curState;
    PlayerState prevState;
    for (const auto& state : curStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.name == nameList.name[whichPlayer]) {
            curState = state;
            found = true;
        }
    }

    if (!found) { // if not present in current tick(not alive)
        selfObs = { 0 };
        return;
    }

    prevState = curState;
    for (const auto& state : prevStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.name == nameList.name[whichPlayer]) {
            prevState = state;
        }
    }

    // TODO normalize all values to -1.0 to 1.0
    // Dust 2
    // x: 1788 -2204
    // y: 3118 -1164
    // z: 252 -131
    selfObs.globalPosX = normalize(curState.originPosition.x, -2204.f, 1788.f);
    selfObs.globalPosY = normalize(curState.originPosition.y, -1164.f, 3118.f);
    selfObs.globalPosZ = normalize(curState.originPosition.z, -131.f, 252.f);
    selfObs.pitch = normalize(curState.viewAngles.pitch, -180.f, 180.f);
    selfObs.yaw = normalize(curState.viewAngles.yaw, -180.f, 180.f);

    Vector3d localVel = getLocalVelocity(curState.absVelocity, curState.viewAngles.yaw);
    selfObs.velocityForward = normalize(localVel.x, -250.f, 250.f);
    selfObs.velocityRight = normalize(localVel.y, -250.f, 250.f);
    selfObs.velocityUp = normalize(localVel.z, -300.f, 300.f);

    //std::cout << "fallVelocity " << curState.fallVelocity << " |velocityUp " << curState.absVelocity.z << std::endl;

    selfObs.isGrounded = curState.onGround;
    selfObs.duckAmount = curState.duckAmount;
    selfObs.healthPercentage = static_cast<float>(curState.health) / 100.0f; // 0.0 to 1.0
    selfObs.ammoPercentage = static_cast<float>(curState.ammoInClip) / 30.0f;   // 0.0 to 1.0
    selfObs.isReloading = curState.isReloading;

}

bool analyzeAction(const std::vector<PlayerState>& curStates, const std::vector<PlayerState>& prevStates, const NameList& nameList, PlayerAction& action, int whichPlayer) {
    // TODO learn which player action to pair with the current tick. should you take the button status of previous, current, or next tick?

    bool found = false;
    PlayerState curState;
    PlayerState prevState;
    for (const auto& state : curStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.name == nameList.name[whichPlayer]) {
            curState = state;
            found = true;
        }
    }

    if (!found) { // if not present in current tick(not alive)
        action = { 0 };
        return false;
    }

    prevState = curState;
    for (const auto& state : prevStates) {
        // TODO, also get the prev state(if born this tick and prev state doesnt exist, copy current) to calculate delta view angles
        if (state.name == nameList.name[whichPlayer]) {
            prevState = state;
        }
    }

    action.deltaPitch = normalizeAngle(curState.viewAngles.pitch - prevState.viewAngles.pitch);
    action.deltaYaw = normalizeAngle(curState.viewAngles.yaw - prevState.viewAngles.yaw);
    // action.deltaPitch = curState.viewAngleVelocity.pitch; // TODO find out if velocity works better than pure angles
    // action.deltaYaw = curState.viewAngleVelocity.yaw;

    uint64_t buttonMask = curState.buttonMask;

    action.isShooting = ((buttonMask & (1ULL << 0)) != 0); // left click
    action.isJumping = ((buttonMask & (1ULL << 1)) != 0);    // Space
    action.isCrouching = ((buttonMask & (1ULL << 2)) != 0);  // Ctrl
    action.isReloading = ((buttonMask & (1ULL << 13)) != 0); // pressing R
    action.isWalking = ((buttonMask & (1ULL << 16)) != 0);    // Shift

    float moveX = 0.f;
    float moveY = 0.f;
    if ((buttonMask & (1ULL << 3)) != 0) {
        moveX += 1.f;
    }
    if ((buttonMask & (1ULL << 4)) != 0) {
        moveX -= 1.f;
    }
    if ((buttonMask & (1ULL << 9)) != 0) {
        moveY -= 1.f;
    }
    if ((buttonMask & (1ULL << 10)) != 0) {
        moveY += 1.f;
    }
    action.moveX = moveX;
    action.moveY = moveY;

    return true;
}

void printPlayerAction(const PlayerAction& action) {

    std::cout << "==============================================================" << std::endl;
    std::cout << action.deltaPitch << "|" <<
        action.deltaYaw << "|" <<
        action.moveX << "|" <<
        action.moveY << "|" <<
        action.isShooting << "|" <<
        action.isJumping << "|" <<
        action.isCrouching << "|" <<
        action.isReloading << "|" <<
        action.isWalking << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
}

void printDamageObservations(const std::vector<DamageObservation>& damageObservations) {

    std::cout << "==============================================================" << std::endl;
    for (auto& damageObservation : damageObservations) {
        std::cout << damageObservation.isValid << "|" <<
            damageObservation.damageDeltaPitch << "|" <<
            damageObservation.damageDeltaYaw << "|" <<
            damageObservation.damageIntensity << "|" <<
            damageObservation.source.x << "|" << damageObservation.source.y << "|" <<
            damageObservation.name.name << std::endl;
    }
    std::cout << "--------------------------------------------------------------" << std::endl;
}

void printSelfObservation(const SelfObservation& selfObservation) {

    std::cout << "==============================================================" << std::endl;
    std::cout << "globalPosX " << selfObservation.globalPosX << "|globalPosY" <<
        selfObservation.globalPosY << "|globalPosZ" <<
        selfObservation.globalPosZ << "|pitch" <<
        selfObservation.pitch << "|yaw" <<
        selfObservation.yaw << "|velocityForward" <<
        selfObservation.velocityForward << "|velocityRight" <<
        selfObservation.velocityRight << "|velocityUp" <<
        selfObservation.velocityUp << "|isGrounded" <<
        selfObservation.isGrounded << "|duckAmount" <<
        selfObservation.duckAmount << "|healthPercentage" <<
        selfObservation.healthPercentage << "|ammoPercentage" <<
        selfObservation.ammoPercentage << "|isReloading" <<
        selfObservation.isReloading << std::endl;
    std::cout << "--------------------------------------------------------------" << std::endl;
}

void printEnemyObservations(const std::vector<EnemyObservation>& enemyObservations) {

    std::cout << "==============================================================" << std::endl;
    for (auto& enemyObservation : enemyObservations) {
        std::cout << enemyObservation.isValid << "|" <<
            enemyObservation.isVisible << "|proximity " <<
            enemyObservation.proximityScore << "|" <<
            enemyObservation.targetDeltaPitch << "|" <<
            enemyObservation.targetDeltaYaw << "|" <<
            enemyObservation.name.name << "|" <<
            enemyObservation.lastSeenPos.x << "|" << enemyObservation.lastSeenPos.y << "|" << enemyObservation.lastSeenPos.z << "|" <<
            enemyObservation.ticksSinceLastSeen << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
    }
}

bool hasPlayerActed(const PlayerAction& action) {
    const float EPSILON = 0.0001f;

    return std::abs(action.deltaPitch) > EPSILON ||
        std::abs(action.deltaYaw) > EPSILON ||
        std::abs(action.moveX) > EPSILON ||
        std::abs(action.moveY) > EPSILON ||
        action.isShooting ||
        action.isJumping ||
        action.isCrouching ||
        action.isReloading ||
        action.isWalking;
}

FlatFrame getFlatFrame(SelfObservation selfObs, std::vector<EnemyObservation> enemyObs , std::vector<DamageObservation> damageObs, PlayerAction action) {
    
    // Check if there is exactly 5 enemyObs and 2 damageObs
    if (enemyObs.size() != 5 || damageObs.size() != 2) {
        FlatFrame emptyFrame;
        std::memset(emptyFrame.values, 0, sizeof(emptyFrame.values));
        return emptyFrame;
    }

    float arr[65];

    // SelfObservation
    arr[0] = selfObs.globalPosX;
    arr[1] = selfObs.globalPosY;
    arr[2] = selfObs.globalPosZ;
    arr[3] = selfObs.pitch;
    arr[4] = selfObs.yaw;
    arr[5] = selfObs.velocityForward;
    arr[6] = selfObs.velocityRight;
    arr[7] = selfObs.velocityUp;
    arr[8] = selfObs.isGrounded;
    arr[9] = selfObs.duckAmount;
    arr[10] = selfObs.healthPercentage;
    arr[11] = selfObs.ammoPercentage;
    arr[12] = selfObs.isReloading;

    // EnemyObservation 0
    arr[13] = enemyObs[0].isValid ? 1.f : 0.f;
    arr[14] = enemyObs[0].isVisible ? 1.f : 0.f;
    arr[15] = enemyObs[0].isImmune ? 1.f : 0.f;
    arr[16] = enemyObs[0].proximityScore;
    arr[17] = enemyObs[0].targetDeltaPitch;
    arr[18] = enemyObs[0].targetDeltaYaw;
    arr[19] = enemyObs[0].memoryConfidence;

    // EnemyObservation 1
    arr[20] = enemyObs[1].isValid ? 1.f : 0.f;
    arr[21] = enemyObs[1].isVisible ? 1.f : 0.f;
    arr[22] = enemyObs[1].isImmune ? 1.f : 0.f;
    arr[23] = enemyObs[1].proximityScore;
    arr[24] = enemyObs[1].targetDeltaPitch;
    arr[25] = enemyObs[1].targetDeltaYaw;
    arr[26] = enemyObs[1].memoryConfidence;

    // EnemyObservation 2
    arr[27] = enemyObs[2].isValid ? 1.f : 0.f;
    arr[28] = enemyObs[2].isVisible ? 1.f : 0.f;
    arr[29] = enemyObs[2].isImmune ? 1.f : 0.f;
    arr[30] = enemyObs[2].proximityScore;
    arr[31] = enemyObs[2].targetDeltaPitch;
    arr[32] = enemyObs[2].targetDeltaYaw;
    arr[33] = enemyObs[2].memoryConfidence;

    // EnemyObservation 3
    arr[34] = enemyObs[3].isValid ? 1.f : 0.f;
    arr[35] = enemyObs[3].isVisible ? 1.f : 0.f;
    arr[36] = enemyObs[3].isImmune ? 1.f : 0.f;
    arr[37] = enemyObs[3].proximityScore;
    arr[38] = enemyObs[3].targetDeltaPitch;
    arr[39] = enemyObs[3].targetDeltaYaw;
    arr[40] = enemyObs[3].memoryConfidence;

    // EnemyObservation 4
    arr[41] = enemyObs[4].isValid ? 1.f : 0.f;
    arr[42] = enemyObs[4].isVisible ? 1.f : 0.f;
    arr[43] = enemyObs[4].isImmune ? 1.f : 0.f;
    arr[44] = enemyObs[4].proximityScore;
    arr[45] = enemyObs[4].targetDeltaPitch;
    arr[46] = enemyObs[4].targetDeltaYaw;
    arr[47] = enemyObs[4].memoryConfidence;

    // DamageObservation 0
    arr[48] = damageObs[0].isValid ? 1.f : 0.f;
    arr[49] = damageObs[0].damageDeltaPitch;
    arr[50] = damageObs[0].damageDeltaYaw;
    arr[51] = damageObs[0].damageIntensity;

    // DamageObservation 1
    arr[52] = damageObs[1].isValid ? 1.f : 0.f;
    arr[53] = damageObs[1].damageDeltaPitch;
    arr[54] = damageObs[1].damageDeltaYaw;
    arr[55] = damageObs[1].damageIntensity;

    // PlayerAction
    arr[56] = action.deltaPitch;
    arr[57] = action.deltaYaw;
    arr[58] = action.moveX;
    arr[59] = action.moveY;
    arr[60] = action.isShooting ? 1.f : 0.f;
    arr[61] = action.isJumping ? 1.f : 0.f;
    arr[62] = action.isCrouching ? 1.f : 0.f;
    arr[63] = action.isReloading ? 1.f : 0.f;
    arr[64] = action.isWalking ? 1.f : 0.f;


    // Package into the struct and return
    FlatFrame result;
    std::memcpy(result.values, arr, sizeof(arr));
    return result;
}

std::string getCsvHeader() {
    std::string header = "";
    std::vector<int> timeSteps = { -16, -12, -8, -4, 0 };

    for (size_t i = 0; i < timeSteps.size(); ++i) {
        // Format prefix as t-16_, t-12_, etc. (handling negative signs cleanly)
        std::string t = "t" + std::to_string(timeSteps[i]) + "_";

        // Self state (13 columns)
        header += t + "self_posX," + t + "self_posY," + t + "self_posZ," +
            t + "self_pitch," + t + "self_yaw," + t + "self_velForward," +
            t + "self_velRight," + t + "self_velUp," + t + "self_isGrounded," +
            t + "self_duckAmount," + t + "self_healthPct," + t + "self_ammoPct," +
            t + "self_isReloading,";

        // Enemy states (5 enemies * 7 columns = 35 columns)
        for (int e = 0; e < 5; ++e) {
            std::string prefix = t + "enemy" + std::to_string(e) + "_";
            header += prefix + "isValid," + prefix + "isVisible," + prefix + "isImmune," +
                prefix + "proxScore," + prefix + "deltaPitch," + prefix + "deltaYaw," +
                prefix + "memoryConf,";
        }

        // Damage states (2 damages * 4 columns = 8 columns)
        for (int d = 0; d < 2; ++d) {
            std::string prefix = t + "damage" + std::to_string(d) + "_";
            header += prefix + "isValid," + prefix + "deltaPitch," +
                prefix + "deltaYaw," + prefix + "intensity,";
        }

        // Action states (9 columns)
        header += t + "action_deltaPitch," + t + "action_deltaYaw," +
            t + "action_moveX," + t + "action_moveY," +
            t + "action_isShooting," + t + "action_isJumping," +
            t + "action_isCrouching," + t + "action_isReloading," +
            t + "action_isWalking";

        // Add a comma separator between timestamp blocks
        if (i < timeSteps.size() - 1) {
            header += ",";
        }
    }

    return header;
}

std::string frameToString(FlatFrame frame) {
    std::string result = "";
    for (int i = 0; i < 65; ++i) {
        if (i > 0) {
            result += ",";
        }
        result += std::to_string(frame.values[i]);
    }
    return result;
}

void processPlayer(const std::string fileName, const uint32_t fromTick, const uint32_t toTick, const NameList nameList, const int whichPlayer, const VisCheck& checker) {
    //uint32_t fromTick = 171;
    //uint32_t toTick = 686;
    std::ifstream file(fileName, std::ios::binary);

    std::vector<PlayerState> prevStates;
    std::vector<PlayerState> curStates;
    std::vector<PlayerState> futureStates;

    uint32_t futureTick = 0;

    std::vector<DamageObservation> damageObservations;
    damageObservations.push_back({ 0 });
    damageObservations.push_back({ 0 });

    std::vector<EnemyObservation> enemyObservations;
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });

    SelfObservation selfObservation = { 0 };
    PlayerAction playerAction = { 0 };

    bool hasMoved = false;

    file.clear();
    file.seekg(0, std::ios::beg);
    std::ofstream outFile;
    std::deque<FlatFrame> historyBuffer;
    PlayerState state;
    while (file.read(
        reinterpret_cast<char*>(&state),
        sizeof(PlayerState)))
    {
        if (state.tick < fromTick || state.tick > toTick) {
            continue;
        }

        if (futureTick == 0) {
            futureTick = state.tick;
        }

        if (state.tick != futureTick) {
            // Process the states for the current tick
            //std::cout << "Processing tick: " << futureTick - 1 << " with " << curStates.size() << " states." << std::endl;

            // if exist current tick but didnt exist in prev, create a new file
            bool wasInPrev = false;
            for (const auto& prevState : prevStates) {
                if (prevState.name == nameList.name[whichPlayer]) {
                    wasInPrev = true;
                }
            }

            bool found = false;
            for (const auto& curState : curStates) {
                if (curState.name == nameList.name[whichPlayer]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (outFile.is_open()) {
                    outFile.close();
                }
            }
            else {
                //TODO analyze and write to file?
                if (!wasInPrev && !outFile.is_open()) {
                    outFile.open(std::string(nameList.name[whichPlayer].name) + std::to_string(curStates[0].tick) + ".bin", std::ios::binary); // TODO  change back to .bin from .csv and put std::ios::binary
                    hasMoved = false;
                    // TODO create a struct for training data, then create The Ring Buffer (double-ended queue) of 16 frames. initially copy current frame to all and then fill in as new frames come till tick-16
                    // std::deque<flatFrame> historyBuffer;
                }

                if (outFile.is_open()) {
                    //outFile << curStates[0].tick << std::endl;
                    analyzeAction(futureStates, curStates, nameList, playerAction, whichPlayer);
                    //printPlayerAction(playerAction);
                    if (!hasMoved) {
                        if (hasPlayerActed(playerAction)) {
                            // TODO populate the history buffer with prevStates data
                            hasMoved = true;

                            analyzeSelf(prevStates, prevStates, nameList, selfObservation, whichPlayer);
                            analyzeEnemy(prevStates, prevStates, nameList, checker, enemyObservations, whichPlayer);
                            analyzeDamage(prevStates, prevStates, nameList, checker, damageObservations, whichPlayer);
                            FlatFrame prevFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, PlayerAction{ 0 });

                            historyBuffer.clear();
                            for (int i = 0; i < 16; i++) {
                                historyBuffer.push_front(prevFrame);
                            }

                            //outFile << getCsvHeader() << std::endl;

                            // reset all observaitons
                            selfObservation = { 0 };
                            std::fill(enemyObservations.begin(), enemyObservations.end(), EnemyObservation{ 0 });
                            std::fill(damageObservations.begin(), damageObservations.end(), DamageObservation{ 0 });
                            //playerAction = { 0 };
                        }
                    }
                    if (hasMoved) {
                        // TODO analyze and record all data
                        analyzeDamage(curStates, prevStates, nameList, checker, damageObservations, whichPlayer);
                        analyzeEnemy(curStates, prevStates, nameList, checker, enemyObservations, whichPlayer);
                        analyzeSelf(curStates, prevStates, nameList, selfObservation, whichPlayer);
                        FlatFrame currentFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, playerAction);

                        outFile.write( // t - 16
                            reinterpret_cast<const char*>(&historyBuffer[15]),
                            sizeof(FlatFrame)
                        );
                        outFile.write( // t - 12
                            reinterpret_cast<const char*>(&historyBuffer[11]),
                            sizeof(FlatFrame)
                        );
                        outFile.write( // t - 8
                            reinterpret_cast<const char*>(&historyBuffer[7]),
                            sizeof(FlatFrame)
                        );
                        outFile.write( // t - 4
                            reinterpret_cast<const char*>(&historyBuffer[3]),
                            sizeof(FlatFrame)
                        );
                        outFile.write( // t - 0 (current)
                            reinterpret_cast<const char*>(&currentFrame),
                            sizeof(FlatFrame)
                        );

                        // put the current frame to history buffer and pop the oldest one
                        //playerAction = // TODO update playerAction
                        currentFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, playerAction);
                        historyBuffer.push_front(currentFrame);
                        historyBuffer.pop_back();

                    }
                }

            }

            // After processing, move current states to previous states and clear current states
            prevStates = std::move(curStates);
            curStates = std::move(futureStates);
            futureStates.clear();
            futureTick = state.tick;
        }

        futureStates.push_back(state);

    }
}

int main() {
    const VisCheck checker = VisCheck::VisCheck("de_dust2.tri");

    std::string fileName = R"(server_states_333_181532.bin)";
    uint32_t fromTick = 350;
    uint32_t toTick = 181527;

    std::ifstream file(fileName, std::ios::binary);

    SteamIDList steamIdList = {};
    NameList nameList = {};

    PlayerState state;

    uint32_t count = 0;
    uint32_t currentTick = 171;
    uint32_t minTick = INT32_MAX;
    uint32_t maxTick = 0;
    uint32_t maxJump = 0;
    uint32_t maxJumpState = 0;
    uint32_t counter = 0;
    uint32_t totalStates = 0;

    //uint32_t tickToInspect = 687; // 686 ferris last alive tick, 687 first dead tick

    while (file.read(
        reinterpret_cast<char*>(&state),
        sizeof(PlayerState)))
    {
        if (state.tick < 171 || state.tick > 38600) {
            continue;
        }

        //if (state.tick == tickToInspect) {
        //    std::cout << "Name: " << state.name.name << std::endl;
        //}

        totalStates++;

        if (state.tick < minTick) {
            minTick = state.tick;
        }
        if (state.tick > maxTick) {
            maxTick = state.tick;
        }
        if (state.tick != currentTick) {
            count++;
            int jump = state.tick - currentTick;
            if (jump > maxJump) {
                maxJump = jump;
                maxJumpState = state.tick;
            }
            currentTick = state.tick;
        }
        if (!inNameList(state.name, nameList)) {

            nameList.name[counter] = state.name;
            counter++;
        }

    }


    std::cout << "Total ticks read: " << count << std::endl;
    std::cout << "Total states read: " << totalStates << std::endl;
    std::cout << "Unique names: " << counter << std::endl;
    std::cout << "Min tick: " << minTick << std::endl;
    std::cout << "Max tick: " << maxTick << std::endl;
    std::cout << "Max jump: " << maxJump << std::endl;
    std::cout << "Max jump state: " << maxJumpState << std::endl;
    std::cout << "names: " << std::endl;
    for (size_t i = 0; i < counter; ++i) {
        std::cout << nameList.name[i].name << std::endl;
    }



    int whichPlayer = 0;
    //processPlayer(fileName, fromTick, toTick, nameList, whichPlayer, checker); // TODO put this on a thread and do synchronous process for 10 players
    std::thread thread_0(processPlayer, fileName, fromTick, toTick, nameList, 0, std::cref(checker));
    std::thread thread_1(processPlayer, fileName, fromTick, toTick, nameList, 1, std::cref(checker));
    std::thread thread_2(processPlayer, fileName, fromTick, toTick, nameList, 2, std::cref(checker));
    std::thread thread_3(processPlayer, fileName, fromTick, toTick, nameList, 3, std::cref(checker));
    std::thread thread_4(processPlayer, fileName, fromTick, toTick, nameList, 4, std::cref(checker));
    std::thread thread_5(processPlayer, fileName, fromTick, toTick, nameList, 5, std::cref(checker));
    std::thread thread_6(processPlayer, fileName, fromTick, toTick, nameList, 6, std::cref(checker));
    std::thread thread_7(processPlayer, fileName, fromTick, toTick, nameList, 7, std::cref(checker));
    std::thread thread_8(processPlayer, fileName, fromTick, toTick, nameList, 8, std::cref(checker));
    std::thread thread_9(processPlayer, fileName, fromTick, toTick, nameList, 9, std::cref(checker));

    thread_0.join();
    thread_1.join();
    thread_2.join();
    thread_3.join();
    thread_4.join();
    thread_5.join();
    thread_6.join();
    thread_7.join();
    thread_8.join();
    thread_9.join();

    return 0;

    std::vector<PlayerState> prevStates; 
    std::vector<PlayerState> curStates;
    std::vector<PlayerState> futureStates;

    uint32_t futureTick = 0;

    std::vector<DamageObservation> damageObservations;
    damageObservations.push_back({ 0 });
    damageObservations.push_back({ 0 });

    std::vector<EnemyObservation> enemyObservations;
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });
    enemyObservations.push_back({ 0 });

    SelfObservation selfObservation = { 0 };
    PlayerAction playerAction = { 0 };

    bool hasMoved = false;

    file.clear();
    file.seekg(0, std::ios::beg);
    std::ofstream outFile;
    std::deque<FlatFrame> historyBuffer;

    while (file.read(
        reinterpret_cast<char*>(&state),
        sizeof(PlayerState)))
    {
        if (state.tick < fromTick || state.tick > toTick) {
            continue;
        }

        if (futureTick == 0) {
            futureTick = state.tick;
        }

        if (state.tick != futureTick) {
            // Process the states for the current tick
            std::cout << "Processing tick: " << futureTick-1 << " with " << curStates.size() << " states." << std::endl;

            // if exist current tick but didnt exist in prev, create a new file
            bool wasInPrev = false;
            for (const auto& prevState : prevStates) {
                if (prevState.name == nameList.name[whichPlayer]) {
                    wasInPrev = true;
                }
            }

            bool found = false;
            for (const auto& curState : curStates) {
                if (curState.name == nameList.name[whichPlayer]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (outFile.is_open()) {
                    outFile.close();
                }
            }
            else {
                //TODO analyze and write to file?
                if (!wasInPrev && !outFile.is_open()) {
                    outFile.open(std::string(nameList.name[whichPlayer].name) + std::to_string(curStates[0].tick) + ".csv"); // TODO  change back to .bin from .csv and put std::ios::binary
                    hasMoved = false;
                    // TODO create a struct for training data, then create The Ring Buffer (double-ended queue) of 16 frames. initially copy current frame to all and then fill in as new frames come till tick-16
                    // std::deque<flatFrame> historyBuffer;
                }

                if (outFile.is_open()) {
                    //outFile << curStates[0].tick << std::endl;
                    analyzeAction(futureStates, curStates, nameList, playerAction, whichPlayer);
                    //printPlayerAction(playerAction);
                    if (!hasMoved) {
                        if (hasPlayerActed(playerAction)) {
                            // TODO populate the history buffer with prevStates data
                            hasMoved = true;

                            analyzeSelf(prevStates, prevStates, nameList, selfObservation, whichPlayer);
                            analyzeEnemy(prevStates, prevStates, nameList, checker, enemyObservations, whichPlayer);
                            analyzeDamage(prevStates, prevStates, nameList, checker, damageObservations, whichPlayer);
                            FlatFrame prevFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, PlayerAction{ 0 });

                            historyBuffer.clear();
                            for (int i = 0; i < 16; i++) {
                                historyBuffer.push_front(prevFrame);
                            }
                            
                            outFile << getCsvHeader() << std::endl;

                            // reset all observaitons
                            selfObservation = { 0 };
                            std::fill(enemyObservations.begin(), enemyObservations.end(), EnemyObservation{0});
                            std::fill(damageObservations.begin(), damageObservations.end(), DamageObservation{ 0 });
                            //playerAction = { 0 };
                        }
                    }
                    if (hasMoved) {
                        // TODO analyze and record all data
                        analyzeDamage(curStates, prevStates, nameList, checker, damageObservations, whichPlayer);
                        analyzeEnemy(curStates, prevStates, nameList, checker, enemyObservations, whichPlayer);
                        analyzeSelf(curStates, prevStates, nameList, selfObservation, whichPlayer);
                        FlatFrame currentFrame = getFlatFrame(selfObservation, enemyObservations, damageObservations, playerAction);

                        //outFile.write( // t - 16
                        //    reinterpret_cast<const char*>(&historyBuffer[15]),
                        //    sizeof(FlatFrame)
                        //);
                        //outFile.write( // t - 12
                        //    reinterpret_cast<const char*>(&historyBuffer[11]),
                        //    sizeof(FlatFrame)
                        //);
                        //outFile.write( // t - 8
                        //    reinterpret_cast<const char*>(&historyBuffer[7]),
                        //    sizeof(FlatFrame)
                        //);
                        //outFile.write( // t - 4
                        //    reinterpret_cast<const char*>(&historyBuffer[3]),
                        //    sizeof(FlatFrame)
                        //);
                        //outFile.write( // t - 0 (current)
                        //    reinterpret_cast<const char*>(&currentFrame),
                        //    sizeof(FlatFrame)
                        //);
                        outFile << frameToString(historyBuffer[15]) << ",";
                        outFile << frameToString(historyBuffer[11]) << ",";
                        outFile << frameToString(historyBuffer[7]) << ",";
                        outFile << frameToString(historyBuffer[3]) << ",";
                        outFile << frameToString(currentFrame) << std::endl;

                        // put the current frame to history buffer and pop the oldest one
                        historyBuffer.push_front(currentFrame);
                        historyBuffer.pop_back();

                    }
                }

            }

            // After processing, move current states to previous states and clear current states
            prevStates = std::move(curStates);
            curStates = std::move(futureStates);
            futureStates.clear();
            futureTick = state.tick;
        }

        futureStates.push_back(state);

    }

    return 0;
}
