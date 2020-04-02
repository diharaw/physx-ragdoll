#pragma once

#include "animation.h"
#include <unordered_set>

struct aiNode;
struct aiBone;
struct aiScene;

struct Joint
{
    std::string name;
    glm::mat4   inverse_bind_pose;
    glm::mat4   offset_from_parent;
    int32_t     parent_index;

    glm::vec3 bind_pos_ws(glm::mat4 model)
    {
        glm::mat4 m = model * glm::inverse(inverse_bind_pose);
        return glm::vec3(m[3][0], m[3][1], m[3][2]);
    }
};

extern glm::mat4 create_offset_transform(glm::mat4 a, glm::mat4 b);

class Skeleton
{
public:
    static Skeleton* create(const aiScene* scene);

    Skeleton();
    ~Skeleton();
    int32_t find_joint_index(const std::string& channel_name);

    inline uint32_t num_bones() { return m_num_joints; }
    inline Joint*   joints() { return &m_joints[0]; }

private:
    void build_bone_list(aiNode* node, const aiScene* scene, std::vector<aiBone*>& temp_bone_list, std::unordered_set<std::string>& bone_map);
    void build_skeleton(aiNode* node, int bone_index, const aiScene* scene, std::vector<aiBone*>& temp_bone_list);

private:
    uint32_t           m_num_joints;
    std::vector<Joint> m_joints;
};