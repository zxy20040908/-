#include "cgltf.h"
#include "meshoptimizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

struct PrimitiveData
{
    std::vector<unsigned int> indices;
    std::vector<float> positions; // xyz
    std::vector<float> normals;   // xyz
    std::vector<float> uv0;       // uv
    int material = -1;
    bool has_normals = false;
    bool has_uv0 = false;
};

struct MeshData
{
    std::vector<PrimitiveData> primitives;
};

struct MaterialData
{
    bool has_name = false;
    std::string name;
};

struct NodeData
{
    bool has_name = false;
    std::string name;

    int mesh = -1;
    std::vector<int> children;

    bool has_matrix = false;
    float matrix[16] = {};

    bool has_translation = false;
    float translation[3] = {};

    bool has_rotation = false;
    float rotation[4] = {};

    bool has_scale = false;
    float scale[3] = {};
};

struct SceneData
{
    std::vector<int> nodes;
};

struct Stats
{
    size_t total_primitives = 0;
    size_t simplified_primitives = 0;
    size_t skipped_primitives = 0;
    size_t failed_primitives = 0;
    size_t no_reduction_primitives = 0;
    size_t input_triangles = 0;
    size_t output_triangles = 0;
    size_t weighted_dihedral_edges = 0;
    size_t weighted_dihedral_vertices = 0;
};

struct AttributeSimplifyConfig
{
    float w_dihedral = 0.0f;
    float dihedral_angle_deg = 35.0f;
};

struct RunConfig
{
    float target_error = 1e-2f;
    size_t min_triangles = 0;
    bool fail_on_no_reduction = false;
    bool verbose = false;
    bool has_report_json = false;
    std::string report_json;
};

struct PrimitiveReport
{
    size_t mesh_index = 0;
    size_t primitive_index = 0;
    size_t triangles_before = 0;
    size_t target_triangles = 0;
    size_t triangles_after = 0;
    size_t dihedral_edges = 0;
    size_t dihedral_vertices = 0;
    std::string status;
    std::string reason;
};

static void append_padded(std::vector<unsigned char>& dst, const void* data, size_t size)
{
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    dst.insert(dst.end(), bytes, bytes + size);
    while ((dst.size() & 3) != 0)
        dst.push_back(0);
}

static bool parse_args(int argc, char** argv, std::string& input, std::string& output, float& ratio)
{
    input.clear();
    output.clear();
    ratio = -1.f;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
            input = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
            ratio = static_cast<float>(atof(argv[++i]));
    }

    if (input.empty() || output.empty() || ratio <= 0.f || ratio > 1.f)
        return false;

    auto has_ext = [](const std::string& s, const char* ext) -> bool
    {
        if (s.size() < strlen(ext))
            return false;

        std::string tail = s.substr(s.size() - strlen(ext));
        for (size_t i = 0; i < tail.size(); ++i)
            tail[i] = char(tolower(static_cast<unsigned char>(tail[i])));
        return tail == ext;
    };

    return has_ext(input, ".glb") && has_ext(output, ".glb");
}

static bool parse_lock_border_arg(int argc, char** argv, unsigned int& simplify_options)
{
    simplify_options = 0;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--lock-border") == 0 || strcmp(argv[i], "-lb") == 0)
        {
            if (i + 1 >= argc)
                return false;

            const int v = atoi(argv[++i]);
            if (v != 0 && v != 1)
                return false;

            if (v == 1)
                simplify_options |= meshopt_SimplifyLockBorder;
        }
    }

    return true;
}

static bool parse_merge_mesh_arg(int argc, char** argv, bool& merge_mesh)
{
    merge_mesh = false;

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--merge-mesh") == 0)
        {
            if (i + 1 >= argc)
                return false;

            const int v = atoi(argv[++i]);
            if (v != 0 && v != 1)
                return false;

            merge_mesh = (v == 1);
        }
    }

    return true;
}

static bool parse_attribute_config_arg(int argc, char** argv, AttributeSimplifyConfig& cfg)
{
    cfg = AttributeSimplifyConfig();

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--w-dihedral") == 0)
        {
            if (i + 1 >= argc)
                return false;
            cfg.w_dihedral = static_cast<float>(atof(argv[++i]));
            if (cfg.w_dihedral < 0.f)
                return false;
        }
        else if (strcmp(argv[i], "--dihedral-angle-deg") == 0)
        {
            if (i + 1 >= argc)
                return false;
            cfg.dihedral_angle_deg = static_cast<float>(atof(argv[++i]));
            if (cfg.dihedral_angle_deg < 0.f || cfg.dihedral_angle_deg > 180.f)
                return false;
        }
        else if (
            strcmp(argv[i], "--w-normal") == 0 ||
            strcmp(argv[i], "--w-uv") == 0 ||
            strcmp(argv[i], "--w-boundary") == 0 ||
            strcmp(argv[i], "--w-feature") == 0 ||
            strcmp(argv[i], "--w-curv") == 0 ||
            strcmp(argv[i], "--w-sil") == 0 ||
            strcmp(argv[i], "--w-thin") == 0 ||
            strcmp(argv[i], "--feature-angle-deg") == 0 ||
            strcmp(argv[i], "--sil-samples") == 0 ||
            strcmp(argv[i], "--boundary-lock") == 0 ||
            strcmp(argv[i], "--feature-lock") == 0)
        {
            // Legacy attribute knobs are intentionally disabled in this mode.
            return false;
        }
    }

    return true;
}

static bool parse_run_config_arg(int argc, char** argv, RunConfig& cfg)
{
    cfg = RunConfig();

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--target-error") == 0)
        {
            if (i + 1 >= argc)
                return false;
            cfg.target_error = static_cast<float>(atof(argv[++i]));
            if (cfg.target_error < 0.f)
                return false;
        }
        else if (strcmp(argv[i], "--min-triangles") == 0)
        {
            if (i + 1 >= argc)
                return false;
            const int v = atoi(argv[++i]);
            if (v < 0)
                return false;
            cfg.min_triangles = static_cast<size_t>(v);
        }
        else if (strcmp(argv[i], "--report-json") == 0)
        {
            if (i + 1 >= argc)
                return false;
            cfg.report_json = argv[++i];
            cfg.has_report_json = !cfg.report_json.empty();
        }
        else if (strcmp(argv[i], "--fail-on-no-reduction") == 0)
        {
            if (i + 1 >= argc)
                return false;
            const int v = atoi(argv[++i]);
            if (v != 0 && v != 1)
                return false;
            cfg.fail_on_no_reduction = (v == 1);
        }
        else if (strcmp(argv[i], "--verbose") == 0)
        {
            if (i + 1 >= argc)
                return false;
            const int v = atoi(argv[++i]);
            if (v != 0 && v != 1)
                return false;
            cfg.verbose = (v == 1);
        }
    }

    return true;
}

static std::vector<unsigned char> compute_feature_flags(
    const std::vector<unsigned int>& indices,
    const std::vector<float>& positions,
    size_t vertex_count,
    float dihedral_angle_deg,
    size_t* out_feature_edges = NULL)
{
    std::vector<unsigned char> feature(vertex_count, 0);
    if (out_feature_edges)
        *out_feature_edges = 0;
    if (indices.size() < 3 || positions.size() < vertex_count * 3)
        return feature;

    struct FaceNormal
    {
        float x, y, z;
        bool valid;
    };

    const size_t tri_count = indices.size() / 3;
    std::vector<FaceNormal> normals(tri_count);

    auto pos = [&positions](unsigned int v) -> const float*
    {
        return &positions[size_t(v) * 3];
    };

    for (size_t t = 0; t < tri_count; ++t)
    {
        const unsigned int a = indices[t * 3 + 0];
        const unsigned int b = indices[t * 3 + 1];
        const unsigned int c = indices[t * 3 + 2];
        if (a >= vertex_count || b >= vertex_count || c >= vertex_count)
        {
            normals[t] = {0.f, 0.f, 0.f, false};
            continue;
        }

        const float* pa = pos(a);
        const float* pb = pos(b);
        const float* pc = pos(c);

        const float abx = pb[0] - pa[0], aby = pb[1] - pa[1], abz = pb[2] - pa[2];
        const float acx = pc[0] - pa[0], acy = pc[1] - pa[1], acz = pc[2] - pa[2];

        const float nx = aby * acz - abz * acy;
        const float ny = abz * acx - abx * acz;
        const float nz = abx * acy - aby * acx;
        const float len2 = nx * nx + ny * ny + nz * nz;
        if (len2 <= 1e-20f)
        {
            normals[t] = {0.f, 0.f, 0.f, false};
            continue;
        }

        const float inv_len = 1.f / std::sqrt(len2);
        normals[t] = {nx * inv_len, ny * inv_len, nz * inv_len, true};
    }

    struct EdgeFace
    {
        unsigned long long key;
        size_t tri;
    };

    std::vector<EdgeFace> ef;
    ef.reserve(indices.size());

    auto edge_key = [](unsigned int a, unsigned int b) -> unsigned long long
    {
        const unsigned int lo = (a < b) ? a : b;
        const unsigned int hi = (a < b) ? b : a;
        return (static_cast<unsigned long long>(lo) << 32) | static_cast<unsigned long long>(hi);
    };

    for (size_t t = 0; t < tri_count; ++t)
    {
        const unsigned int a = indices[t * 3 + 0];
        const unsigned int b = indices[t * 3 + 1];
        const unsigned int c = indices[t * 3 + 2];
        if (a >= vertex_count || b >= vertex_count || c >= vertex_count)
            continue;

        ef.push_back({edge_key(a, b), t});
        ef.push_back({edge_key(b, c), t});
        ef.push_back({edge_key(c, a), t});
    }

    std::sort(ef.begin(), ef.end(), [](const EdgeFace& lhs, const EdgeFace& rhs)
    {
        return lhs.key < rhs.key;
    });

    const float angle_threshold_rad = dihedral_angle_deg * (3.14159265358979323846f / 180.f);
    const float cos_threshold = std::cos(angle_threshold_rad);

    size_t i = 0;
    while (i < ef.size())
    {
        size_t j = i + 1;
        while (j < ef.size() && ef[j].key == ef[i].key)
            ++j;

        const unsigned int lo = static_cast<unsigned int>(ef[i].key >> 32);
        const unsigned int hi = static_cast<unsigned int>(ef[i].key & 0xffffffffull);

        bool mark = false;
        if (j - i > 2)
        {
            mark = true; // non-manifold edge: protect conservatively
        }
        else if (j - i == 2)
        {
            const FaceNormal& n0 = normals[ef[i + 0].tri];
            const FaceNormal& n1 = normals[ef[i + 1].tri];
            if (n0.valid && n1.valid)
            {
                float d = n0.x * n1.x + n0.y * n1.y + n0.z * n1.z;
                if (d < -1.f) d = -1.f;
                if (d > 1.f) d = 1.f;
                if (d <= cos_threshold)
                    mark = true;
            }
        }

        if (mark)
        {
            if (out_feature_edges)
                *out_feature_edges += 1;
            if (lo < vertex_count)
                feature[lo] = 1;
            if (hi < vertex_count)
                feature[hi] = 1;
        }

        i = j;
    }

    return feature;
}

static std::vector<float> remap_stream(const std::vector<float>& src, size_t components, const std::vector<unsigned int>& remap, size_t dst_count)
{
    std::vector<float> dst(dst_count * components, 0.f);
    const size_t src_count = remap.size();

    for (size_t i = 0; i < src_count; ++i)
    {
        const unsigned int r = remap[i];
        if (r == ~0u)
            continue;

        for (size_t c = 0; c < components; ++c)
            dst[r * components + c] = src[i * components + c];
    }

    return dst;
}

static std::string json_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);

    for (size_t i = 0; i < s.size(); ++i)
    {
        const char c = s[i];
        if (c == '\\')
            out += "\\\\";
        else if (c == '"')
            out += "\\\"";
        else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c == '\t')
            out += "\\t";
        else
            out += c;
    }

    return out;
}

static bool write_report_json(
    const char* path,
    const char* input_path,
    const char* output_path,
    float ratio,
    unsigned int simplify_options,
    bool merge_mesh,
    const AttributeSimplifyConfig& attr_cfg,
    const RunConfig& run_cfg,
    const Stats& stats,
    const std::vector<PrimitiveReport>& reports)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return false;

    std::string json = "{";
    json += "\"input\":\"" + json_escape(input_path ? input_path : "") + "\",";
    json += "\"output\":\"" + json_escape(output_path ? output_path : "") + "\",";
    json += "\"ratio\":" + std::to_string(ratio) + ",";
    json += "\"options\":{";
    json += "\"lock_border\":" + std::to_string((simplify_options & meshopt_SimplifyLockBorder) ? 1 : 0) + ",";
    json += "\"merge_mesh\":" + std::to_string(merge_mesh ? 1 : 0) + ",";
    json += "\"w_dihedral\":" + std::to_string(attr_cfg.w_dihedral) + ",";
    json += "\"dihedral_angle_deg\":" + std::to_string(attr_cfg.dihedral_angle_deg) + ",";
    json += "\"target_error\":" + std::to_string(run_cfg.target_error) + ",";
    json += "\"min_triangles\":" + std::to_string(run_cfg.min_triangles) + ",";
    json += "\"fail_on_no_reduction\":" + std::to_string(run_cfg.fail_on_no_reduction ? 1 : 0) + ",";
    json += "\"verbose\":" + std::to_string(run_cfg.verbose ? 1 : 0);
    json += "},";

    json += "\"stats\":{";
    json += "\"total_primitives\":" + std::to_string(stats.total_primitives) + ",";
    json += "\"simplified_primitives\":" + std::to_string(stats.simplified_primitives) + ",";
    json += "\"skipped_primitives\":" + std::to_string(stats.skipped_primitives) + ",";
    json += "\"failed_primitives\":" + std::to_string(stats.failed_primitives) + ",";
    json += "\"no_reduction_primitives\":" + std::to_string(stats.no_reduction_primitives) + ",";
    json += "\"input_triangles\":" + std::to_string(stats.input_triangles) + ",";
    json += "\"output_triangles\":" + std::to_string(stats.output_triangles) + ",";
    json += "\"weighted_dihedral_edges\":" + std::to_string(stats.weighted_dihedral_edges) + ",";
    json += "\"weighted_dihedral_vertices\":" + std::to_string(stats.weighted_dihedral_vertices);
    json += "},";

    json += "\"primitives\":[";
    for (size_t i = 0; i < reports.size(); ++i)
    {
        if (i)
            json += ",";
        const PrimitiveReport& r = reports[i];
        json += "{";
        json += "\"mesh_index\":" + std::to_string(r.mesh_index) + ",";
        json += "\"primitive_index\":" + std::to_string(r.primitive_index) + ",";
        json += "\"triangles_before\":" + std::to_string(r.triangles_before) + ",";
        json += "\"target_triangles\":" + std::to_string(r.target_triangles) + ",";
        json += "\"triangles_after\":" + std::to_string(r.triangles_after) + ",";
        json += "\"dihedral_edges\":" + std::to_string(r.dihedral_edges) + ",";
        json += "\"dihedral_vertices\":" + std::to_string(r.dihedral_vertices) + ",";
        json += "\"status\":\"" + json_escape(r.status) + "\",";
        json += "\"reason\":\"" + json_escape(r.reason) + "\"";
        json += "}";
    }
    json += "]";
    json += "}";

    const size_t written = fwrite(json.data(), 1, json.size(), f);
    fclose(f);
    return written == json.size();
}

struct EmittedPrimitive
{
    size_t pos_offset = 0;
    size_t pos_size = 0;
    size_t nor_offset = 0;
    size_t nor_size = 0;
    size_t uv_offset = 0;
    size_t uv_size = 0;
    size_t idx_offset = 0;
    size_t idx_size = 0;

    size_t vertex_count = 0;
    size_t index_count = 0;

    bool has_normals = false;
    bool has_uv0 = false;

    int material = -1;

    int pos_bv = -1;
    int nor_bv = -1;
    int uv_bv = -1;
    int idx_bv = -1;

    int pos_acc = -1;
    int nor_acc = -1;
    int uv_acc = -1;
    int idx_acc = -1;

    float minv[3] = {};
    float maxv[3] = {};
};

static bool primitive_output_valid(const PrimitiveData& p)
{
    return !p.positions.empty() && !p.indices.empty() && (p.positions.size() % 3 == 0) && (p.indices.size() % 3 == 0);
}

static bool write_glb_geometry(
    const char* path,
    const std::vector<MeshData>& meshes,
    const std::vector<MaterialData>& materials,
    const std::vector<NodeData>& nodes,
    const std::vector<SceneData>& scenes,
    int default_scene,
    bool merge_mesh)
{
    // Keep only meshes with valid primitives and build remap old->new.
    std::vector<int> mesh_map(meshes.size(), -1);
    std::vector<int> exported_mesh_ids;

    for (size_t i = 0; i < meshes.size(); ++i)
    {
        bool ok = false;
        for (size_t p = 0; p < meshes[i].primitives.size(); ++p)
            ok = ok || primitive_output_valid(meshes[i].primitives[p]);

        if (ok)
        {
            mesh_map[i] = static_cast<int>(exported_mesh_ids.size());
            exported_mesh_ids.push_back(static_cast<int>(i));
        }
    }

    if (exported_mesh_ids.empty())
        return false;

    std::vector<unsigned char> bin;
    std::vector<std::vector<EmittedPrimitive> > emitted(exported_mesh_ids.size());

    int next_bv = 0;
    int next_acc = 0;

    for (size_t mi = 0; mi < exported_mesh_ids.size(); ++mi)
    {
        const MeshData& mesh = meshes[exported_mesh_ids[mi]];

        for (size_t pi = 0; pi < mesh.primitives.size(); ++pi)
        {
            const PrimitiveData& p = mesh.primitives[pi];
            if (!primitive_output_valid(p))
                continue;

            EmittedPrimitive e;
            e.vertex_count = p.positions.size() / 3;
            e.index_count = p.indices.size();
            e.has_normals = p.has_normals && p.normals.size() == e.vertex_count * 3;
            e.has_uv0 = p.has_uv0 && p.uv0.size() == e.vertex_count * 2;
            e.material = p.material;

            e.pos_offset = bin.size();
            e.pos_size = p.positions.size() * sizeof(float);
            append_padded(bin, p.positions.data(), e.pos_size);

            e.minv[0] = e.maxv[0] = p.positions[0];
            e.minv[1] = e.maxv[1] = p.positions[1];
            e.minv[2] = e.maxv[2] = p.positions[2];

            for (size_t v = 0; v < e.vertex_count; ++v)
            {
                const float* pp = &p.positions[v * 3];
                e.minv[0] = std::min(e.minv[0], pp[0]);
                e.minv[1] = std::min(e.minv[1], pp[1]);
                e.minv[2] = std::min(e.minv[2], pp[2]);
                e.maxv[0] = std::max(e.maxv[0], pp[0]);
                e.maxv[1] = std::max(e.maxv[1], pp[1]);
                e.maxv[2] = std::max(e.maxv[2], pp[2]);
            }

            e.pos_bv = next_bv++;
            e.pos_acc = next_acc++;

            if (e.has_normals)
            {
                e.nor_offset = bin.size();
                e.nor_size = p.normals.size() * sizeof(float);
                append_padded(bin, p.normals.data(), e.nor_size);
                e.nor_bv = next_bv++;
                e.nor_acc = next_acc++;
            }

            if (e.has_uv0)
            {
                e.uv_offset = bin.size();
                e.uv_size = p.uv0.size() * sizeof(float);
                append_padded(bin, p.uv0.data(), e.uv_size);
                e.uv_bv = next_bv++;
                e.uv_acc = next_acc++;
            }

            e.idx_offset = bin.size();
            e.idx_size = p.indices.size() * sizeof(unsigned int);
            append_padded(bin, p.indices.data(), e.idx_size);
            e.idx_bv = next_bv++;
            e.idx_acc = next_acc++;

            emitted[mi].push_back(e);
        }
    }

    std::string json = "{";
    json += "\"asset\":{\"version\":\"2.0\",\"generator\":\"puresimplify\"},";
    json += "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) + "}],";

    // bufferViews
    json += "\"bufferViews\":[";
    bool first = true;

    auto emit_bv = [&](size_t offset, size_t size, int target)
    {
        if (!first)
            json += ",";
        first = false;

        json += "{\"buffer\":0,\"byteOffset\":" + std::to_string(offset) + ",\"byteLength\":" + std::to_string(size) + ",\"target\":" + std::to_string(target) + "}";
    };

    for (size_t mi = 0; mi < emitted.size(); ++mi)
    {
        for (size_t pi = 0; pi < emitted[mi].size(); ++pi)
        {
            const EmittedPrimitive& e = emitted[mi][pi];
            emit_bv(e.pos_offset, e.pos_size, 34962);
            if (e.has_normals)
                emit_bv(e.nor_offset, e.nor_size, 34962);
            if (e.has_uv0)
                emit_bv(e.uv_offset, e.uv_size, 34962);
            emit_bv(e.idx_offset, e.idx_size, 34963);
        }
    }
    json += "],";

    // accessors
    json += "\"accessors\":[";
    first = true;

    auto emit_accessor = [&](int view, int component_type, size_t count, const char* type)
    {
        if (!first)
            json += ",";
        first = false;
        json += "{\"bufferView\":" + std::to_string(view) + ",\"byteOffset\":0,\"componentType\":" + std::to_string(component_type) + ",\"count\":" + std::to_string(count) + ",\"type\":\"" + type + "\"}";
    };

    for (size_t mi = 0; mi < emitted.size(); ++mi)
    {
        for (size_t pi = 0; pi < emitted[mi].size(); ++pi)
        {
            const EmittedPrimitive& e = emitted[mi][pi];

            if (!first)
                json += ",";
            first = false;
            json += "{\"bufferView\":" + std::to_string(e.pos_bv) + ",\"byteOffset\":0,\"componentType\":5126,\"count\":" + std::to_string(e.vertex_count) + ",\"type\":\"VEC3\",\"min\":[" +
                    std::to_string(e.minv[0]) + "," + std::to_string(e.minv[1]) + "," + std::to_string(e.minv[2]) +
                    "],\"max\":[" + std::to_string(e.maxv[0]) + "," + std::to_string(e.maxv[1]) + "," + std::to_string(e.maxv[2]) + "]}";

            if (e.has_normals)
                emit_accessor(e.nor_bv, 5126, e.vertex_count, "VEC3");
            if (e.has_uv0)
                emit_accessor(e.uv_bv, 5126, e.vertex_count, "VEC2");
            emit_accessor(e.idx_bv, 5125, e.index_count, "SCALAR");
        }
    }
    json += "],";

    // materials
    if (!materials.empty())
    {
        json += "\"materials\":[";
        for (size_t i = 0; i < materials.size(); ++i)
        {
            if (i)
                json += ",";

            if (materials[i].has_name)
                json += "{\"name\":\"" + json_escape(materials[i].name) + "\"}";
            else
                json += "{}";
        }
        json += "],";
    }

    // meshes
    json += "\"meshes\":[";
    if (merge_mesh)
    {
        json += "{\"primitives\":[";
        bool prim_first = true;

        for (size_t mi = 0; mi < emitted.size(); ++mi)
        {
            for (size_t pi = 0; pi < emitted[mi].size(); ++pi)
            {
                const EmittedPrimitive& e = emitted[mi][pi];
                if (!prim_first)
                    json += ",";
                prim_first = false;

                json += "{\"attributes\":{\"POSITION\":" + std::to_string(e.pos_acc);
                if (e.has_normals)
                    json += ",\"NORMAL\":" + std::to_string(e.nor_acc);
                if (e.has_uv0)
                    json += ",\"TEXCOORD_0\":" + std::to_string(e.uv_acc);
                json += "},\"indices\":" + std::to_string(e.idx_acc) + ",\"mode\":4";

                if (e.material >= 0 && size_t(e.material) < materials.size())
                    json += ",\"material\":" + std::to_string(e.material);

                json += "}";
            }
        }

        json += "]}";
    }
    else
    {
        for (size_t mi = 0; mi < emitted.size(); ++mi)
        {
            if (mi)
                json += ",";

            json += "{\"primitives\":[";
            for (size_t pi = 0; pi < emitted[mi].size(); ++pi)
            {
                const EmittedPrimitive& e = emitted[mi][pi];
                if (pi)
                    json += ",";

                json += "{\"attributes\":{\"POSITION\":" + std::to_string(e.pos_acc);
                if (e.has_normals)
                    json += ",\"NORMAL\":" + std::to_string(e.nor_acc);
                if (e.has_uv0)
                    json += ",\"TEXCOORD_0\":" + std::to_string(e.uv_acc);
                json += "},\"indices\":" + std::to_string(e.idx_acc) + ",\"mode\":4";

                if (e.material >= 0 && size_t(e.material) < materials.size())
                    json += ",\"material\":" + std::to_string(e.material);

                json += "}";
            }
            json += "]}";
        }
    }
    json += "],";

    if (merge_mesh)
    {
        json += "\"nodes\":[{\"mesh\":0}],";
        json += "\"scenes\":[{\"nodes\":[0]}],";
        json += "\"scene\":0";
    }
    else
    {
        // nodes
        json += "\"nodes\":[";
        for (size_t ni = 0; ni < nodes.size(); ++ni)
        {
            if (ni)
                json += ",";

            const NodeData& n = nodes[ni];
            json += "{";
            bool need = false;

            if (n.has_name)
            {
                json += "\"name\":\"" + json_escape(n.name) + "\"";
                need = true;
            }

            if (n.mesh >= 0 && size_t(n.mesh) < mesh_map.size() && mesh_map[n.mesh] >= 0)
            {
                if (need)
                    json += ",";
                json += "\"mesh\":" + std::to_string(mesh_map[n.mesh]);
                need = true;
            }

            if (!n.children.empty())
            {
                if (need)
                    json += ",";
                json += "\"children\":[";
                for (size_t i = 0; i < n.children.size(); ++i)
                {
                    if (i)
                        json += ",";
                    json += std::to_string(n.children[i]);
                }
                json += "]";
                need = true;
            }

            if (n.has_matrix)
            {
                if (need)
                    json += ",";
                json += "\"matrix\":[";
                for (int i = 0; i < 16; ++i)
                {
                    if (i)
                        json += ",";
                    json += std::to_string(n.matrix[i]);
                }
                json += "]";
                need = true;
            }
            else
            {
                if (n.has_translation)
                {
                    if (need)
                        json += ",";
                    json += "\"translation\":[" + std::to_string(n.translation[0]) + "," + std::to_string(n.translation[1]) + "," + std::to_string(n.translation[2]) + "]";
                    need = true;
                }

                if (n.has_rotation)
                {
                    if (need)
                        json += ",";
                    json += "\"rotation\":[" + std::to_string(n.rotation[0]) + "," + std::to_string(n.rotation[1]) + "," + std::to_string(n.rotation[2]) + "," + std::to_string(n.rotation[3]) + "]";
                    need = true;
                }

                if (n.has_scale)
                {
                    if (need)
                        json += ",";
                    json += "\"scale\":[" + std::to_string(n.scale[0]) + "," + std::to_string(n.scale[1]) + "," + std::to_string(n.scale[2]) + "]";
                    need = true;
                }
            }

            json += "}";
        }
        json += "],";

        // scenes
        json += "\"scenes\":[";
        for (size_t si = 0; si < scenes.size(); ++si)
        {
            if (si)
                json += ",";
            json += "{\"nodes\":[";
            for (size_t i = 0; i < scenes[si].nodes.size(); ++i)
            {
                if (i)
                    json += ",";
                json += std::to_string(scenes[si].nodes[i]);
            }
            json += "]}";
        }
        json += "]";

        if (default_scene >= 0 && size_t(default_scene) < scenes.size())
            json += ",\"scene\":" + std::to_string(default_scene);
    }

    json += "}";

    std::vector<unsigned char> json_chunk(json.begin(), json.end());
    while ((json_chunk.size() & 3) != 0)
        json_chunk.push_back(' ');

    const uint32_t json_length = static_cast<uint32_t>(json_chunk.size());
    const uint32_t bin_length = static_cast<uint32_t>(bin.size());
    const uint32_t total_length = 12 + 8 + json_length + 8 + bin_length;

    FILE* f = fopen(path, "wb");
    if (!f)
        return false;

    const uint32_t magic = 0x46546C67u;
    const uint32_t version = 2;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&total_length, 4, 1, f);

    const uint32_t json_type = 0x4E4F534Au;
    fwrite(&json_length, 4, 1, f);
    fwrite(&json_type, 4, 1, f);
    fwrite(json_chunk.data(), 1, json_chunk.size(), f);

    const uint32_t bin_type = 0x004E4942u;
    fwrite(&bin_length, 4, 1, f);
    fwrite(&bin_type, 4, 1, f);
    fwrite(bin.data(), 1, bin.size(), f);

    fclose(f);
    return true;
}

static bool simplify_glb(
    const char* input_path,
    float ratio,
    unsigned int simplify_options,
    const AttributeSimplifyConfig& attr_cfg,
    const RunConfig& run_cfg,
    std::vector<MeshData>& out_meshes,
    std::vector<MaterialData>& out_materials,
    std::vector<NodeData>& out_nodes,
    std::vector<SceneData>& out_scenes,
    int& out_default_scene,
    Stats& stats,
    std::vector<PrimitiveReport>& primitive_reports)
{
    cgltf_options options = {};
    cgltf_data* data = NULL;

    cgltf_result res = cgltf_parse_file(&options, input_path, &data);
    if (res != cgltf_result_success)
    {
        std::printf("error: cgltf_parse_file failed (%d)\n", int(res));
        return false;
    }

    res = cgltf_load_buffers(&options, data, input_path);
    if (res != cgltf_result_success)
    {
        std::printf("error: cgltf_load_buffers failed (%d)\n", int(res));
        cgltf_free(data);
        return false;
    }

    if (data->file_type != cgltf_file_type_glb)
    {
        std::printf("error: input is not glb\n");
        cgltf_free(data);
        return false;
    }

    // Copy scene graph metadata.
    out_meshes.clear();
    out_meshes.resize(data->meshes_count);

    out_materials.clear();
    out_materials.resize(data->materials_count);
    for (cgltf_size i = 0; i < data->materials_count; ++i)
    {
        if (data->materials[i].name)
        {
            out_materials[i].has_name = true;
            out_materials[i].name = data->materials[i].name;
        }
    }

    out_nodes.clear();
    out_nodes.resize(data->nodes_count);
    for (cgltf_size i = 0; i < data->nodes_count; ++i)
    {
        const cgltf_node& n = data->nodes[i];
        NodeData nd;

        if (n.name)
        {
            nd.has_name = true;
            nd.name = n.name;
        }

        nd.mesh = n.mesh ? int(n.mesh - data->meshes) : -1;

        nd.children.resize(n.children_count);
        for (cgltf_size c = 0; c < n.children_count; ++c)
            nd.children[c] = int(n.children[c] - data->nodes);

        nd.has_matrix = n.has_matrix != 0;
        nd.has_translation = n.has_translation != 0;
        nd.has_rotation = n.has_rotation != 0;
        nd.has_scale = n.has_scale != 0;

        memcpy(nd.matrix, n.matrix, sizeof(nd.matrix));
        memcpy(nd.translation, n.translation, sizeof(nd.translation));
        memcpy(nd.rotation, n.rotation, sizeof(nd.rotation));
        memcpy(nd.scale, n.scale, sizeof(nd.scale));

        out_nodes[i] = nd;
    }

    out_scenes.clear();
    out_scenes.resize(data->scenes_count);
    for (cgltf_size i = 0; i < data->scenes_count; ++i)
    {
        out_scenes[i].nodes.resize(data->scenes[i].nodes_count);
        for (cgltf_size n = 0; n < data->scenes[i].nodes_count; ++n)
            out_scenes[i].nodes[n] = int(data->scenes[i].nodes[n] - data->nodes);
    }

    out_default_scene = data->scene ? int(data->scene - data->scenes) : -1;

    primitive_reports.clear();

    auto append_output_primitive = [&](size_t mesh_index,
                                       const cgltf_primitive& prim,
                                       const std::vector<unsigned int>& indices,
                                       const std::vector<float>& positions,
                                       const std::vector<float>& normals,
                                       const std::vector<float>& uv0,
                                       bool has_nor,
                                       bool has_uv0)
    {
        PrimitiveData out;
        out.indices = indices;
        out.positions = positions;
        out.material = prim.material ? int(prim.material - data->materials) : -1;
        out.has_normals = has_nor;
        out.has_uv0 = has_uv0;
        if (has_nor)
            out.normals = normals;
        if (has_uv0)
            out.uv0 = uv0;
        out_meshes[mesh_index].primitives.push_back(std::move(out));
    };

    // Simplify mesh primitives.
    for (cgltf_size mi = 0; mi < data->meshes_count; ++mi)
    {
        const cgltf_mesh& mesh = data->meshes[mi];
        for (cgltf_size pi = 0; pi < mesh.primitives_count; ++pi)
        {
            const cgltf_primitive& prim = mesh.primitives[pi];
            PrimitiveReport report = {};
            report.mesh_index = size_t(mi);
            report.primitive_index = size_t(pi);
            stats.total_primitives++;

            if (prim.type != cgltf_primitive_type_triangles || !prim.indices)
            {
                stats.skipped_primitives++;
                std::printf("warning: skip non-triangle primitive m%zu p%zu\n", size_t(mi), size_t(pi));
                report.status = "failed_invalid_output";
                report.reason = "non_triangle_primitive";
                primitive_reports.push_back(report);
                continue;
            }

            const cgltf_accessor* pos = NULL;
            const cgltf_accessor* nor = NULL;
            const cgltf_accessor* uv0 = NULL;

            for (cgltf_size ai = 0; ai < prim.attributes_count; ++ai)
            {
                const cgltf_attribute& attr = prim.attributes[ai];
                if (attr.type == cgltf_attribute_type_position)
                    pos = attr.data;
                else if (attr.type == cgltf_attribute_type_normal)
                    nor = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0)
                    uv0 = attr.data;
            }

            if (!pos)
            {
                stats.skipped_primitives++;
                std::printf("warning: skip primitive without POSITION m%zu p%zu\n", size_t(mi), size_t(pi));
                report.status = "failed_invalid_output";
                report.reason = "missing_position";
                primitive_reports.push_back(report);
                continue;
            }

            const size_t index_count = prim.indices->count;
            if (index_count < 3 || (index_count % 3) != 0)
            {
                stats.skipped_primitives++;
                std::printf("warning: skip invalid index count m%zu p%zu\n", size_t(mi), size_t(pi));
                report.status = "failed_invalid_output";
                report.reason = "invalid_index_count";
                primitive_reports.push_back(report);
                continue;
            }

            report.triangles_before = index_count / 3;
            report.triangles_after = report.triangles_before;
            stats.input_triangles += report.triangles_before;

            const size_t src_vertex_count = pos->count;
            std::vector<unsigned int> src_indices(index_count);
            for (size_t i = 0; i < index_count; ++i)
                src_indices[i] = static_cast<unsigned int>(cgltf_accessor_read_index(prim.indices, i));

            std::vector<float> src_positions(src_vertex_count * 3);
            for (size_t i = 0; i < src_vertex_count; ++i)
                cgltf_accessor_read_float(pos, i, &src_positions[i * 3], 3);

            bool has_nor = false;
            std::vector<float> src_normals;
            if (nor && nor->count == src_vertex_count)
            {
                has_nor = true;
                src_normals.resize(src_vertex_count * 3);
                for (size_t i = 0; i < src_vertex_count; ++i)
                    cgltf_accessor_read_float(nor, i, &src_normals[i * 3], 3);
            }

            bool has_uv0 = false;
            std::vector<float> src_uv0;
            if (uv0 && uv0->count == src_vertex_count)
            {
                has_uv0 = true;
                src_uv0.resize(src_vertex_count * 2);
                for (size_t i = 0; i < src_vertex_count; ++i)
                    cgltf_accessor_read_float(uv0, i, &src_uv0[i * 2], 2);
            }

            // Weld by position + optional normal/uv0 streams.
            std::vector<unsigned int> remap(src_vertex_count);
            std::vector<meshopt_Stream> streams;

            meshopt_Stream s0 = {src_positions.data(), sizeof(float) * 3, sizeof(float) * 3};
            streams.push_back(s0);

            if (has_nor)
            {
                meshopt_Stream sn = {src_normals.data(), sizeof(float) * 3, sizeof(float) * 3};
                streams.push_back(sn);
            }

            if (has_uv0)
            {
                meshopt_Stream st = {src_uv0.data(), sizeof(float) * 2, sizeof(float) * 2};
                streams.push_back(st);
            }

            const size_t welded_vertex_count = meshopt_generateVertexRemapMulti(remap.data(), src_indices.data(), index_count, src_vertex_count, streams.data(), streams.size());

            std::vector<unsigned int> welded_indices(index_count);
            meshopt_remapIndexBuffer(welded_indices.data(), src_indices.data(), index_count, remap.data());

            std::vector<float> welded_positions = remap_stream(src_positions, 3, remap, welded_vertex_count);
            std::vector<float> welded_normals;
            std::vector<float> welded_uv0;

            if (has_nor)
                welded_normals = remap_stream(src_normals, 3, remap, welded_vertex_count);
            if (has_uv0)
                welded_uv0 = remap_stream(src_uv0, 2, remap, welded_vertex_count);

            size_t target_index_count = static_cast<size_t>(index_count * ratio);
            target_index_count = (target_index_count / 3) * 3;
            target_index_count = std::max<size_t>(3, target_index_count);
            target_index_count = std::min(target_index_count, index_count);
            if (run_cfg.min_triangles > 0)
                target_index_count = std::max(target_index_count, run_cfg.min_triangles * 3);
            target_index_count = std::min(target_index_count, index_count);
            report.target_triangles = target_index_count / 3;

            size_t dihedral_edge_count = 0;
            std::vector<unsigned char> dihedral_flags;
            if (attr_cfg.w_dihedral > 0.f)
            {
                dihedral_flags = compute_feature_flags(
                    welded_indices,
                    welded_positions,
                    welded_vertex_count,
                    attr_cfg.dihedral_angle_deg,
                    &dihedral_edge_count);
            }

            size_t dihedral_vertex_count = 0;
            for (size_t v = 0; v < dihedral_flags.size(); ++v)
                dihedral_vertex_count += (dihedral_flags[v] != 0);

            report.dihedral_edges = dihedral_edge_count;
            report.dihedral_vertices = dihedral_vertex_count;
            if (attr_cfg.w_dihedral > 0.f)
            {
                stats.weighted_dihedral_edges += dihedral_edge_count;
                stats.weighted_dihedral_vertices += dihedral_vertex_count;
            }

            std::vector<unsigned int> simplified(index_count);
            size_t simplified_count = 0;
            if (attr_cfg.w_dihedral > 0.f)
            {
                std::vector<float> attr_buffer(welded_vertex_count, 0.f);
                for (size_t v = 0; v < welded_vertex_count; ++v)
                    attr_buffer[v] = dihedral_flags[v] ? 1.f : 0.f;

                float attr_weight = attr_cfg.w_dihedral;
                simplified_count = meshopt_simplifyWithAttributes(
                    simplified.data(),
                    welded_indices.data(),
                    index_count,
                    welded_positions.data(),
                    welded_vertex_count,
                    sizeof(float) * 3,
                    attr_buffer.data(),
                    sizeof(float),
                    &attr_weight,
                    1,
                    NULL,
                    target_index_count,
                    run_cfg.target_error,
                    simplify_options,
                    NULL);
            }
            else
            {
                simplified_count = meshopt_simplify(
                    simplified.data(),
                    welded_indices.data(),
                    index_count,
                    welded_positions.data(),
                    welded_vertex_count,
                    sizeof(float) * 3,
                    target_index_count,
                    run_cfg.target_error,
                    simplify_options,
                    NULL);
            }

            if (simplified_count < 3 || (simplified_count % 3) != 0)
            {
                stats.failed_primitives++;
                if (run_cfg.verbose)
                    std::printf("warning: simplify failed m%zu p%zu\n", size_t(mi), size_t(pi));
                append_output_primitive(size_t(mi), prim, welded_indices, welded_positions, welded_normals, welded_uv0, has_nor, has_uv0);
                stats.output_triangles += report.triangles_before;
                report.status = "failed_optimizer";
                report.reason = "invalid_simplified_count";
                primitive_reports.push_back(report);
                continue;
            }

            simplified.resize(simplified_count);
            report.triangles_after = simplified_count / 3;

            if (simplified_count >= index_count)
            {
                append_output_primitive(size_t(mi), prim, welded_indices, welded_positions, welded_normals, welded_uv0, has_nor, has_uv0);
                stats.no_reduction_primitives++;
                stats.output_triangles += report.triangles_before;
                report.status = "ok_no_reduction";
                report.reason = "no_triangle_reduction";
                primitive_reports.push_back(report);
                continue;
            }

            // Compact used vertices.
            std::vector<unsigned int> compact_remap(welded_vertex_count, ~0u);
            unsigned int next = 0;
            for (size_t i = 0; i < simplified.size(); ++i)
            {
                const unsigned int idx = simplified[i];
                unsigned int& r = compact_remap[idx];
                if (r == ~0u)
                    r = next++;
                simplified[i] = r;
            }

            std::vector<float> compact_positions = remap_stream(welded_positions, 3, compact_remap, next);
            std::vector<float> compact_normals;
            std::vector<float> compact_uv0;

            if (has_nor)
                compact_normals = remap_stream(welded_normals, 3, compact_remap, next);
            if (has_uv0)
                compact_uv0 = remap_stream(welded_uv0, 2, compact_remap, next);

            if (compact_positions.empty() || simplified.empty())
            {
                stats.failed_primitives++;
                append_output_primitive(size_t(mi), prim, welded_indices, welded_positions, welded_normals, welded_uv0, has_nor, has_uv0);
                stats.output_triangles += report.triangles_before;
                report.status = "failed_invalid_output";
                report.reason = "empty_compacted_output";
                primitive_reports.push_back(report);
                continue;
            }

            append_output_primitive(size_t(mi), prim, simplified, compact_positions, compact_normals, compact_uv0, has_nor, has_uv0);

            stats.simplified_primitives++;
            stats.output_triangles += simplified_count / 3;
            report.status = "ok_reduced";
            report.reason = "reduced";
            primitive_reports.push_back(report);
        }
    }

    cgltf_free(data);

    bool has_output = false;
    for (size_t i = 0; i < out_meshes.size(); ++i)
        has_output = has_output || !out_meshes[i].primitives.empty();

    return has_output;
}

int main(int argc, char** argv)
{
    std::string input;
    std::string output;
    float ratio = 0.f;
    unsigned int simplify_options = 0;
    bool merge_mesh = false;
    AttributeSimplifyConfig attr_cfg;
    RunConfig run_cfg;

    if (!parse_args(argc, argv, input, output, ratio))
    {
        std::printf("usage: app -i <input.glb> -o <output.glb> -r <ratio(0,1]> [--lock-border 0|1] [--merge-mesh 0|1] [--w-dihedral f] [--dihedral-angle-deg f] [--target-error f] [--min-triangles n] [--report-json path] [--fail-on-no-reduction 0|1] [--verbose 0|1]\n");
        return 1;
    }

    if (!parse_lock_border_arg(argc, argv, simplify_options))
    {
        std::printf("error: --lock-border (or -lb) must be 0 or 1\n");
        return 1;
    }

    if (!parse_merge_mesh_arg(argc, argv, merge_mesh))
    {
        std::printf("error: --merge-mesh must be 0 or 1\n");
        return 1;
    }

    if (!parse_attribute_config_arg(argc, argv, attr_cfg))
    {
        std::printf("error: invalid attribute options; use only --w-dihedral >= 0 and --dihedral-angle-deg in [0,180]. legacy attribute args are disabled.\n");
        return 1;
    }

    if (!parse_run_config_arg(argc, argv, run_cfg))
    {
        std::printf("error: invalid run options; expected --target-error >= 0, --min-triangles >= 0, optional --report-json path, and 0|1 toggles\n");
        return 1;
    }

    Stats stats;
    std::vector<MeshData> meshes;
    std::vector<MaterialData> materials;
    std::vector<NodeData> nodes;
    std::vector<SceneData> scenes;
    std::vector<PrimitiveReport> primitive_reports;
    int default_scene = -1;

    if (!simplify_glb(input.c_str(), ratio, simplify_options, attr_cfg, run_cfg, meshes, materials, nodes, scenes, default_scene, stats, primitive_reports))
        return 2;

    if (!write_glb_geometry(output.c_str(), meshes, materials, nodes, scenes, default_scene, merge_mesh))
    {
        std::printf("error: write output failed\n");
        return 3;
    }

    if (run_cfg.has_report_json)
    {
        if (!write_report_json(run_cfg.report_json.c_str(), input.c_str(), output.c_str(), ratio, simplify_options, merge_mesh, attr_cfg, run_cfg, stats, primitive_reports))
        {
            std::printf("error: write report json failed\n");
            return 4;
        }
    }

    std::printf("done: primitives total=%zu simplified=%zu skipped=%zu\n", stats.total_primitives, stats.simplified_primitives, stats.skipped_primitives);
    std::printf("triangles: input=%zu output=%zu ratio=%.4f\n", stats.input_triangles, stats.output_triangles,
        stats.input_triangles ? float(stats.output_triangles) / float(stats.input_triangles) : 1.f);
    std::printf("weighted_dihedral: edges=%zu vertices=%zu\n", stats.weighted_dihedral_edges, stats.weighted_dihedral_vertices);
    std::printf("status: failed=%zu no_reduction=%zu\n", stats.failed_primitives, stats.no_reduction_primitives);
    std::printf("options: lock_border=%d\n", (simplify_options & meshopt_SimplifyLockBorder) ? 1 : 0);
    std::printf("options: merge_mesh=%d\n", merge_mesh ? 1 : 0);
    std::printf("options: w_dihedral=%.3f dihedral_angle_deg=%.1f\n", attr_cfg.w_dihedral, attr_cfg.dihedral_angle_deg);
    std::printf("options: target_error=%.6f min_triangles=%zu fail_on_no_reduction=%d verbose=%d\n", run_cfg.target_error, run_cfg.min_triangles, run_cfg.fail_on_no_reduction ? 1 : 0, run_cfg.verbose ? 1 : 0);
    if (run_cfg.has_report_json)
        std::printf("report: %s\n", run_cfg.report_json.c_str());
    std::printf("wrote: %s\n", output.c_str());

    if (run_cfg.fail_on_no_reduction && stats.no_reduction_primitives > 0)
        return 5;

    return 0;
}
