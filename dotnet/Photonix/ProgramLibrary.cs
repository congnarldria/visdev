using System.IO;

namespace Photonix;

sealed record ProgramEntry(string Name, string Title, string Path, string Kind);

sealed record TemplateEntry(string Title, string File, string Expect);

static class ProgramLibrary
{
    public static readonly TemplateEntry[] Templates =
    [
        new("零件檢測", "inspect_parts.viz", "8 個零件"),
        new("邊緣量測", "inspect_gauge.viz", "邊緣點在行 107.5"),
        new("卡尺", "inspect_caliper.viz", "直徑約 83.93，寬 140，高 90"),
        new("瑕疵", "inspect_defect.viz", "暗塊 1 個、面積 3550；亮點 1 個、面積 144"),
        new("色彩", "inspect_color.viz", "綠塊面積 1200、色調 60"),
        new("字元", "inspect_chars.viz", "讀出 VISDEV 168"),
        new("條紋寬度", "inspect_stripe.viz", "最寬亮條 22"),
        new("產線", "inspect_line.viz", "合格 3、不合格 2"),
        new("產線相機", "inspect_line_camera.viz", "相機編號 0。沒有相機時改跑檔案產線"),
        new("產線輸出", "inspect_line_out.viz", "TCP loopback，合格 3、不合格 2"),
        new("缺件", "inspect_missing.viz", "應有 8 個零件"),
        new("孔數", "inspect_holes.viz", "4 個暗孔，直徑約 28，偏差 0"),
        new("孔數產線", "inspect_line_holes.viz", "合格 2、不合格 3"),
        new("形狀定位", "inspect_locate.viz", "場景裡找到 4 個"),
        new("相對姿態", "inspect_stereo.viz", "Pose1 的第一個數約 0.04 公尺"),
        new("條碼", "inspect_code.viz", "Text 是 4006381333931，QrText 是 VISDEV-3"),
        new("找線找圓", "inspect_fit.viz", "直線到行 30，圓心列 50 行 80，半徑 20"),
        new("次像素邊緣", "inspect_edge.viz", "階躍邊緣在行 15.5"),
    ];

    static readonly string[] SampleOrder =
    [
        "blob_inspect.hdv",
        "roi_check.hdv",
        "auto_threshold.hdv",
        "morph_region.hdv",
        "region_sets.hdv",
        "subpix_blob.hdv",
        "edge_amp.hdv",
        "xld_fit.hdv",
        "xld_circle.hdv",
        "image_filter.hdv",
        "tuple_stats.hdv",
        "ViZion_rgb1_to_gray_tuple_add_shift_local_threshold.viz",
        "ViZion_rgb1_to_gray_measure_metrology_scaled_shape.viz",
        "ViZion_create_matrix_rigid_barcode_text.viz",
        "ViZion_open_framegrabber_file_camera_Code39_quaternion_cloud.viz",
        "ViZion_gray_erosion_rect_ellipse_contrast.viz",
        "ViZion_get_image_size_gray_tophat.viz",
        "ViZion_get_grayval_ramp_diameter_shape.viz",
        "ViZion_scale_image_range_project_arrow.viz",
        "ViZion_distance_rr_align_corner_region_distance.viz",
        "ViZion_gen_region_points_power_ridge_intersection.viz",
        "ViZion_get_region_runs_filter_skeleton_circle.viz",
        "ViZion_inner_rectangle1_holes_rounded.viz",
        "ViZion_check_difference_chars_corners.viz",
        "ViZion_gen_empty_region_edge_plateau_flow.viz",
        "ViZion_distance_transform_highpass_corner.viz",
        "ViZion_eliminate_runs_plane_entropy.viz",
        "ViZion_sqrt_image_log_saddle_trim.viz",
        "ViZion_gray_features_distance_auto_threshold.viz",
        "ViZion_gamma_image_zero_cross_sector.viz",
        "ViZion_exp_image_polar_tophat.viz",
        "ViZion_reduce_domain_paint_corner_intersection.viz",
        "ViZion_abs_image_direction_crossing.viz",
        "ViZion_prewitt_amp_polar_split.viz",
        "ViZion_frei_amp_robinson_azimuth_circle.viz",
        "ViZion_bit_not_lut_bits_pyramid.viz",
        "ViZion_gen_measure_rectangle2_profile_funct1d.viz",
        "ViZion_create_funct_1d_pairs_integral_derivative_inverse.viz",
        "ViZion_fft_image_shift_phase_correlation.viz",
        "ViZion_fft_image_spectrum_mask_convolve_energy.viz",
        "ViZion_fft_image_correlation_shift_pad.viz",
        "ViZion_gen_object_model_3d_from_points_plane_box.viz",
        "ViZion_gen_object_model_3d_from_points_scale_project_axis.viz",
        "ViZion_gen_object_model_3d_from_points_intersect_grid_axis.viz",
        "ViZion_gen_object_model_3d_from_region_cloud_aligned_to_image.viz",
        "ViZion_fit_image_plane_height_normal_shade.viz",
        "ViZion_sub_plane_image_reference_tilt_volume.viz",
        "ViZion_fit_plane_region_plane_inside_region.viz",
        "ViZion_fit_plane_objects_plane_per_object.viz",
        "ViZion_fit_quad_region_parabola_quadric.viz",
        "ViZion_add_plane_image_height_map_difference.viz",
        "ViZion_sub_local_mean_local_mean_roughness.viz",
        "ViZion_sub_local_median_median_local_bump.viz",
        "ViZion_gray_spot_white_spots_smaller_than_kernel.viz",
        "ViZion_gray_opening_rect_opening_reconstruct.viz",
        "ViZion_thickness_max_how_thick.viz",
        "ViZion_area_intersection_overlap_area.viz",
        "ViZion_distance_objects_how_far_apart.viz",
        "ViZion_region_angle_which_side.viz",
        "ViZion_partition_nearest_gap_to_nearest.viz",
        "ViZion_sort_feature_pick_by_feature.viz",
        "ViZion_region_defects_fill_notches.viz",
        "ViZion_select_inner_who_contains_whom.viz",
        "ViZion_select_border_touches_border.viz",
        "ViZion_neighbor_distance_objects_touch.viz",
        "ViZion_shape_score_most_like_template.viz",
        "ViZion_align_delta_move_onto_template.viz",
        "ViZion_grid_size_rows_and_columns.viz",
        "ViZion_gray_offset_brightness_side.viz",
        "ViZion_edge_contrast_edge_clarity.viz",
        "ViZion_edge_min_weakest_edge.viz",
        "ViZion_band_count_bright_enough.viz",
        "ViZion_gray_mode_main_gray.viz",
        "ViZion_row_runs_horizontal_and_vertical.viz",
        "ViZion_symmetry_score_left_right_symmetry.viz",
        "ViZion_top_point_four_endpoints.viz",
        "ViZion_hole_count_hole_count_and_size.viz",
        "ViZion_window_inside_stays_inside_window.viz",
        "ViZion_major_phi_major_axis_direction.viz",
        "ViZion_edge_enhance_sharpen_edges_keep_flat.viz",
        "ViZion_area_center_catch_error_after_procedure.viz",
        "ViZion_diff_region_difference_from_golden.viz",
        "ViZion_gen_qrcode_print_qr_read_back.viz",
        "ViZion_ridge_region_dark_scratch_direction.viz",
        "ViZion_edge_region_edge_between_pixels.viz",
        "ViZion_phase_correlation_shift_under_one_pixel.viz",
        "ViZion_gen_contour_polygon_xld_line_outlier_stays_straight.viz",
        "ViZion_gen_contour_polygon_xld_arc_outlier.viz",
        "ViZion_gen_contour_region_xld_boundary_at_half_pixel.viz",
        "ViZion_gen_contour_polygon_xld_gauss_weight_ignores_outlier.viz",
        "ViZion_derivate_gauss_slope1_derivative_is_129.viz",
        "ViZion_get_grayval_interpolated_half_pixel_is_not_the_average.viz",
        "ViZion_teach_shape_model_least_squares_aligns_shape.viz",
        "ViZion_gauss_image_least_squares_row_near_39.25.viz",
        "ViZion_create_metrology_model_caliper_averages_spike.viz",
        "ViZion_points_foerstner_corner_inside_pixel.viz",
        "ViZion_vector_to_hom_mat2d_affine_ignores_outlier.viz",
        "ViZion_hom_vector_to_proj_hom_mat2d_projective_ignores_outlier.viz",
        "ViZion_gen_ellipse_contour_xld_ellipse_arc_outlier.viz",
        "ViZion_tuple_concat_rectangle_edge_outlier.viz",
        "ViZion_fit_plane_region_spike_does_not_tilt_plane.viz",
        "ViZion_gen_object_model_3d_from_points_cloud_plane_ignores_outlier.viz",
        "ViZion_gen_object_model_3d_from_points_axis_ignores_side_point.viz",
        "ViZion_gen_contour_polygon_xld_line_normal_ignores_corner.viz",
        "ViZion_points_harris_harris_corner_inside_pixel.viz",
        "ViZion_gen_contour_polygon_xld_smooth_keeps_endpoints.viz",
        "ViZion_zoom_image_size_zoomed_half_pixel_is_112.viz",
        "ViZion_elliptic_axis_wide_bar_0_tall_bar_90.viz",
        "ViZion_hom_mat2d_identity_pulled_back_half_pixel_is_112.viz",
        "ViZion_local_max_sub_pix_gauss_peak_at_30.33298.viz",
        "ViZion_points_lepetit_square_corner_inside_pixel.viz",
        "ViZion_create_scaled_shape_model_scale_refined_to_1.08895.viz",
        "ViZion_expand_region_region_splits_overlap.viz",
        "ViZion_gen_image_const_real_zero.viz",
        "ViZion_convert_image_type_real_byte_roundtrip.viz",
        "ViZion_gen_image_surface_real_keeps_310.viz",
        "ViZion_find_calib_object_three_views_one_board.viz",
        "ViZion_set_calib_data_observ_points_image_size_is_extent.viz",
        "ViZion_gen_code93_lowercase_reads_VISDEV.viz",
        "ViZion_gen_code2of5_odd_length_reads_012345.viz",
        "ViZion_gen_code128_digits_use_code_set_c.viz",
        "ViZion_get_text_result_blank_image_num_is_0.viz",
        "ViZion_gauss_image_smoothed_shift_row_is_39.23.viz",
        "ViZion_gen_datamatrix_print_datamatrix_read_VISDEV.viz",
        "ViZion_gen_contour_polygon_xld_contour_tuple_stays_hobject.viz",
        "ViZion_gen_pdf417_print_pdf417_read_VISDEV.viz",
        "ViZion_gen_aztec_print_aztec_read_VISDEV.viz",
        "ViZion_open_serial_serial_write_read_VISDEV.viz",
        "ViZion_open_socket_connect_tcp_send_read_VISDEV.viz",
        "ViZion_split_skeleton_cross_into_four.viz",
        "ViZion_set_serial_param_serial_baud_19200.viz",
        "ViZion_open_socket_accept_listen_read_VISDEV.viz",
        "ViZion_gen_ean8_print_ean8_read_55123453.viz",
        "ViZion_rectangularity_bar_rectangularity_and_direction.viz",
        "ViZion_circularity_xld_square_length_is_120.viz",
        "ViZion_remove_obj_two_blocks_drop_second.viz",
        "ViZion_bit_mask_40_and_240_is_32.viz",
        "ViZion_gen_sphere_object_model_3d_center_123_radius_4.viz",
        "ViZion_frangi_dark_line_brighter_than_block.viz",
        "ViZion_sauvola_bright_side_scratch_stays.viz",
        "ViZion_guided_filter_step_sides_stay.viz",
        "ViZion_emphasize_clahe_dark_corner_and_bright_field.viz",
        "ViZion_edges_zernike_edge_matches_measure.viz",
        "ViZion_tuple_is_handle_matrix_stays_after_clear.viz",
        "ViZion_sum_matrix_copy_cholesky_and_file.viz",
        "ViZion_sin_image_gray0_sine_is_128.viz",
        "ViZion_gray_dilation_shape_neighbor_is_9.viz",
        "ViZion_polar_trans_image_radius3_is_200.viz",
        "ViZion_equ_histo_image_rect_lower_right_is_255.viz",
        "ocr_read.hdv",
        "shape_match.hdv",
        "shape_teach.hdv",
        "ncc_locate.hdv",
        "gauge.hdv",
    ];

    public static List<ProgramEntry> Samples(string repo) => List(Path.Combine(repo, "samples"), "samples", true);

    public static List<ProgramEntry> Workspace(string repo) => List(Path.Combine(repo, "workspace"), "workspace", false);

    public static List<ProgramEntry> List(string folder, string kind, bool useSampleOrder)
    {
        var files = new Dictionary<string, string>(StringComparer.Ordinal);
        if (Directory.Exists(folder))
        {
            foreach (string path in Directory.EnumerateFiles(folder))
            {
                string name = Path.GetFileName(path);
                if (name.Length == 0 || name[0] == '.')
                    continue;
                string lower = name.ToLowerInvariant();
                if (!lower.EndsWith(".hdv") && !lower.EndsWith(".viz"))
                    continue;
                files[name] = path;
            }
        }
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        if (useSampleOrder)
        {
            foreach (string name in SampleOrder)
            {
                if (files.ContainsKey(name) && seen.Add(name))
                    names.Add(name);
            }
        }
        foreach (string name in files.Keys.OrderBy(name => name, StringComparer.Ordinal))
        {
            if (seen.Add(name))
                names.Add(name);
        }
        return names.Select(name => new ProgramEntry(name, TitleOf(files[name]), files[name], kind)).ToList();
    }

    public static string TitleOf(string path)
    {
        try
        {
            foreach (string raw in File.ReadLines(path))
            {
                string text = raw.Trim();
                if (text.StartsWith('*'))
                {
                    string title = text.TrimStart('*').Trim();
                    return title.Length == 0 ? Path.GetFileNameWithoutExtension(path) : title;
                }
                if (text.Length > 0 && !text.StartsWith("//"))
                    break;
            }
        }
        catch (Exception)
        {
            // 讀不到就用檔名。
        }
        return Path.GetFileNameWithoutExtension(path);
    }
}
