#include <chrono>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/image.hpp"

using namespace std::chrono_literals;

class DirectCameraLaneFollowerRecorder : public rclcpp::Node {
public:
  // ===========================================================================
  // States
  // ===========================================================================
  enum class State {
    LINE_FOLLOW,
    ODOM_CURVE,
    REACQUIRE_LINE,
    STOP_SIGN,
    GIVE_AWAY,
    LEFT_SIGN,
    RIGHT_SIGN,
    FORWARD_SIGN,
    TRAFFIC_RED_WAIT,
    INTERSECTION_FORWARD,
    INTERSECTION_SPIN,
    ROADWORK,
    WORKERS_AHEAD
  };

  struct LaneLine {
    int x{0}, y{0}, w{1}, h{1};
    double cx{0.0}, cy{0.0};
    int x1{0}, y1{0}, x2{0}, y2{0};
    double angle{0.0}, length{0.0};
  };

  struct HoughCandidate {
    double x_at_scan{0.0};
    int x1{0}, y1{0}, x2{0}, y2{0};
    double angle{0.0}, length{0.0};
  };

  // ===========================================================================
  // Constructor
  // ===========================================================================
  DirectCameraLaneFollowerRecorder()
  : Node("direct_camera_lane_follower_recorder"),
    last_sign_time_(0, 0, RCL_ROS_TIME)
  {
    declareParameters();
    getParameters();

    left_pub_  = create_publisher<std_msgs::msg::Float32>("/VelocitySetL", 10);
    right_pub_ = create_publisher<std_msgs::msg::Float32>("/VelocitySetR", 10);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&DirectCameraLaneFollowerRecorder::odomCallback, this, std::placeholders::_1));

    sign_sub_ = create_subscription<std_msgs::msg::String>(
      "/sign_detected", 10,
      std::bind(&DirectCameraLaneFollowerRecorder::signCallback, this, std::placeholders::_1));

    if (use_image_topic_) {
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&DirectCameraLaneFollowerRecorder::imageCallback, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "Using shared camera topic: %s", image_topic_.c_str());
    } else {
      openCamera();
      if (!cap_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Could not open camera.");
        stopRobot();
        throw std::runtime_error("Camera failed");
      }
    }

    if (seconds_ > 0.0)
      end_time_ = now() + rclcpp::Duration::from_seconds(seconds_);

    const auto period_ms = static_cast<int>(1000.0 / std::max(control_hz_, 1.0));
    timer_ = create_wall_timer(
      std::chrono::milliseconds(period_ms),
      std::bind(&DirectCameraLaneFollowerRecorder::controlStep, this));

    normal_max_speed_ = max_speed_;
    disabled_startup_stop_remaining_ =
      enable_wheel_commands_ ? 0 : std::max(0, disabled_startup_stop_frames_);

    RCLCPP_INFO(get_logger(), "Lane follower CENTER_ZEBRA_UNDISTORT_V4 started.");
    RCLCPP_INFO(get_logger(), "curve_trigger_error_px=%.1f  curve_duration_s=%.2f",
      curve_trigger_error_px_, curve_duration_s_);
    RCLCPP_INFO(get_logger(), "Camera input: %s",
      use_image_topic_ ? image_topic_.c_str() : device_.c_str());
    RCLCPP_INFO(get_logger(), "Wheel commands: %s",
      enable_wheel_commands_ ? "ENABLED" : "DISABLED (monitor mode)");
    if (!enable_wheel_commands_ && disabled_startup_stop_remaining_ > 0) {
      RCLCPP_WARN(get_logger(),
        "Wheel commands disabled: sending startup stop for %d frames to clear latched motor command.",
        disabled_startup_stop_remaining_);
    }
    RCLCPP_INFO(get_logger(), "Debug video: %s", out_path_.c_str());
  }

  ~DirectCameraLaneFollowerRecorder() override {
    stopRobot();
    if (cap_.isOpened())    cap_.release();
    if (writer_.isOpened()) {
      writer_.release();
      RCLCPP_INFO(get_logger(), "Saved video: %s", out_path_.c_str());
    }
  }

private:
  // ===========================================================================
  // Parameter declaration
  // ===========================================================================
  void declareParameters() {
    declare_parameter<bool>("gst", true);
    declare_parameter<std::string>("device", "/dev/video0");
    declare_parameter<int>("source", 0);
    declare_parameter<int>("width", 640);
    declare_parameter<int>("height", 480);
    declare_parameter<int>("fps", 30);
    declare_parameter<double>("control_hz", 15.0);
    declare_parameter<bool>("use_image_topic", false);
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<bool>("enable_wheel_commands", true);
    declare_parameter<int>("disabled_startup_stop_frames", 12);

    // Camera calibration / lens flattening.
    // These values remove radial/tangential distortion before any lane/zebra detection.
    declare_parameter<bool>("enable_undistort", true);
    declare_parameter<int>("calib_width", 1280);
    declare_parameter<int>("calib_height", 720);
    declare_parameter<double>("undistort_alpha", 0.0);

    declare_parameter<std::string>("out", "/home/puzzlebot/lane_drive_debug.avi");
    declare_parameter<double>("video_fps", 15.0);
    declare_parameter<bool>("side_by_side", true);
    declare_parameter<double>("seconds", 20.0);

    declare_parameter<int>("adaptive_block", 41);
    declare_parameter<int>("adaptive_c", 7);
    declare_parameter<int>("black_ceiling", 115);
    declare_parameter<double>("scan_y", 0.72);
    declare_parameter<int>("band_height", 18);
    declare_parameter<int>("hough_threshold", 18);
    declare_parameter<int>("hough_min_length", 45);
    declare_parameter<int>("hough_max_gap", 12);
    declare_parameter<double>("hough_min_abs_angle_deg", 20.0);
    declare_parameter<double>("group_gap", 45.0);
    declare_parameter<int>("fallback_offset_px", 160);
    declare_parameter<int>("deadband_px", 25);
    declare_parameter<int>("max_lost_frames", 30);

    declare_parameter<double>("max_speed", 2.5);
    declare_parameter<double>("speed_straight", 1.1);
    declare_parameter<double>("speed_soft_turn", 0.85);
    declare_parameter<double>("speed_hard_turn", 0.76);
    declare_parameter<double>("search_speed", 0.6);
    declare_parameter<double>("kp_straight", 0.0005);
    declare_parameter<double>("kp_soft_turn", 0.0015);
    declare_parameter<double>("kp_hard_turn", 0.01);
    declare_parameter<double>("small_error_px", 35.0);
    declare_parameter<double>("medium_error_px", 90.0);
    declare_parameter<double>("turn_sign", 1.0);
    declare_parameter<double>("left_motor_sign", 1.0);
    declare_parameter<double>("right_motor_sign", 1.0);

    declare_parameter<bool>("enable_odom_curves", true);
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<double>("curve_radius_m", 0.25);
    declare_parameter<double>("curve_angle_deg", 90.0);
    declare_parameter<double>("curve_linear_speed", 0.18);
    declare_parameter<double>("wheel_base_m", 0.19);
    declare_parameter<double>("min_curve_wheel_speed", 0.12);
    declare_parameter<double>("curve_direction_override_error_px", 45.0);
    declare_parameter<double>("line_follow_correction_limit", 0.70);

    // Umbral de error para disparar ODOM_CURVE automáticamente.
    // Default 65 px: entra a curva antes de que el robot se salga demasiado.
    // No es un clamp del error — el error real sigue siendo el de visión.
    declare_parameter<double>("curve_trigger_error_px", 65.0);
    declare_parameter<int>("curve_trigger_frames", 1);
    declare_parameter<double>("curve_cooldown_s", 1.0);

    // Duración de la curva basada en tiempo de pared (reemplaza dead reckoning).
    // El robot gira durante curve_duration_s_ segundos antes de pasar a REACQUIRE.
    declare_parameter<double>("curve_duration_s", 1.8);

    declare_parameter<double>("reacquire_speed", 0.08);
    declare_parameter<int>("reacquire_frames_needed", 5);
    declare_parameter<double>("reacquire_timeout_s", 3.0);

    declare_parameter<bool>("enable_crossing_gate", true);
    declare_parameter<double>("crossing_roi_y1", 0.55);
    declare_parameter<double>("crossing_roi_y2", 0.90);
    declare_parameter<int>("crossing_min_area", 4500);
    declare_parameter<int>("crossing_min_width_px", 180);
    declare_parameter<int>("crossing_min_height_px", 25);
    declare_parameter<double>("crossing_min_fill_ratio", 0.22);
    declare_parameter<double>("crossing_max_fill_ratio", 0.85);
    declare_parameter<double>("crossing_horizontal_ratio", 2.5);
    declare_parameter<int>("crossing_clear_frames_needed", 4);
    declare_parameter<bool>("crossing_requires_three_lines", true);

    // New camera angle: follow only the center lane marker.
    // Side lane markers are ignored for centering and used only to infer turn direction.
    declare_parameter<bool>("use_center_lane_only", true);
    declare_parameter<double>("center_lane_x_min", 0.34);
    declare_parameter<double>("center_lane_x_max", 0.66);

    // Zebra/stop-line ROI. This is the lower-middle band where the dashed
    // zebra bars appear with the new downward camera angle.
    declare_parameter<bool>("enable_zebra_stop_roi", true);
    declare_parameter<double>("zebra_roi_x1", 0.03);
    declare_parameter<double>("zebra_roi_x2", 0.97);
    declare_parameter<double>("zebra_roi_y1", 0.60);
    declare_parameter<double>("zebra_roi_y2", 0.78);
    declare_parameter<int>("zebra_min_blobs", 3);
    declare_parameter<int>("zebra_frames_needed", 1);
    declare_parameter<int>("zebra_min_area", 250);
    declare_parameter<int>("zebra_min_width_px", 35);
    declare_parameter<int>("zebra_min_height_px", 6);
    declare_parameter<int>("zebra_max_height_px", 45);
    declare_parameter<double>("zebra_min_horizontal_ratio", 2.4);
    declare_parameter<double>("zebra_min_fill_ratio", 0.25);

    declare_parameter<bool>("stop_at_zebra_enabled", true);
    declare_parameter<bool>("give_away_stops_at_zebra", false);
    declare_parameter<double>("stop_zebra_arm_timeout_s", 6.0);

    declare_parameter<double>("sign_timeout_s", 3.0);
    declare_parameter<double>("stop_duration_s", 2.0);
    declare_parameter<double>("give_away_speed_factor", 0.5);
    declare_parameter<double>("slow_speed_factor", 0.5);
    declare_parameter<double>("traffic_green_speed_multiplier", 2.0);
    declare_parameter<double>("traffic_yellow_speed_multiplier", 1.0);
    declare_parameter<double>("traffic_light_timeout_s", 8.0);
    declare_parameter<double>("intersection_action_timeout_s", 8.0);
    declare_parameter<double>("intersection_entry_speed", 0.65);
    declare_parameter<double>("intersection_entry_duration_s", 0.75);
    declare_parameter<double>("intersection_forward_duration_s", 1.25);
    declare_parameter<double>("intersection_spin_speed", 0.55);
    declare_parameter<double>("intersection_spin_duration_s", 0.85);

    // FORWARD: avanza recto durante forward_duration_s_ segundos.
    // No usa curva — elimina la dependencia de last_non_center_direction_.
    declare_parameter<double>("forward_duration_s", 1.5);
  }

  // ===========================================================================
  // Parameter retrieval
  // ===========================================================================
  void getParameters() {
    gst_        = get_parameter("gst").as_bool();
    device_     = get_parameter("device").as_string();
    source_     = get_parameter("source").as_int();
    width_      = get_parameter("width").as_int();
    height_     = get_parameter("height").as_int();
    fps_        = get_parameter("fps").as_int();
    control_hz_ = get_parameter("control_hz").as_double();
    use_image_topic_ = get_parameter("use_image_topic").as_bool();
    image_topic_ = get_parameter("image_topic").as_string();
    enable_wheel_commands_ = get_parameter("enable_wheel_commands").as_bool();
    disabled_startup_stop_frames_ = get_parameter("disabled_startup_stop_frames").as_int();

    enable_undistort_ = get_parameter("enable_undistort").as_bool();
    calib_width_      = get_parameter("calib_width").as_int();
    calib_height_     = get_parameter("calib_height").as_int();
    undistort_alpha_  = get_parameter("undistort_alpha").as_double();

    out_path_    = get_parameter("out").as_string();
    video_fps_   = get_parameter("video_fps").as_double();
    side_by_side_= get_parameter("side_by_side").as_bool();
    seconds_     = get_parameter("seconds").as_double();

    adaptive_block_          = get_parameter("adaptive_block").as_int();
    adaptive_c_              = get_parameter("adaptive_c").as_int();
    black_ceiling_           = get_parameter("black_ceiling").as_int();
    scan_y_norm_             = get_parameter("scan_y").as_double();
    band_height_             = get_parameter("band_height").as_int();
    hough_threshold_         = get_parameter("hough_threshold").as_int();
    hough_min_length_        = get_parameter("hough_min_length").as_int();
    hough_max_gap_           = get_parameter("hough_max_gap").as_int();
    hough_min_abs_angle_deg_ = get_parameter("hough_min_abs_angle_deg").as_double();
    group_gap_               = get_parameter("group_gap").as_double();
    fallback_offset_px_      = get_parameter("fallback_offset_px").as_int();
    deadband_px_             = get_parameter("deadband_px").as_int();
    max_lost_frames_         = get_parameter("max_lost_frames").as_int();

    max_speed_        = get_parameter("max_speed").as_double();
    speed_straight_   = get_parameter("speed_straight").as_double();
    speed_soft_turn_  = get_parameter("speed_soft_turn").as_double();
    speed_hard_turn_  = get_parameter("speed_hard_turn").as_double();
    search_speed_     = get_parameter("search_speed").as_double();
    kp_straight_      = get_parameter("kp_straight").as_double();
    kp_soft_turn_     = get_parameter("kp_soft_turn").as_double();
    kp_hard_turn_     = get_parameter("kp_hard_turn").as_double();
    small_error_px_   = get_parameter("small_error_px").as_double();
    medium_error_px_  = get_parameter("medium_error_px").as_double();
    turn_sign_        = get_parameter("turn_sign").as_double();
    left_motor_sign_  = get_parameter("left_motor_sign").as_double();
    right_motor_sign_ = get_parameter("right_motor_sign").as_double();

    enable_odom_curves_     = get_parameter("enable_odom_curves").as_bool();
    odom_topic_             = get_parameter("odom_topic").as_string();
    curve_radius_m_         = get_parameter("curve_radius_m").as_double();
    curve_angle_deg_        = get_parameter("curve_angle_deg").as_double();
    curve_linear_speed_     = get_parameter("curve_linear_speed").as_double();
    wheel_base_m_           = get_parameter("wheel_base_m").as_double();
    min_curve_wheel_speed_  = get_parameter("min_curve_wheel_speed").as_double();
    curve_direction_override_error_px_ = get_parameter("curve_direction_override_error_px").as_double();
    line_follow_correction_limit_ = get_parameter("line_follow_correction_limit").as_double();
    curve_trigger_error_px_ = get_parameter("curve_trigger_error_px").as_double();
    curve_trigger_frames_   = get_parameter("curve_trigger_frames").as_int();
    curve_cooldown_s_       = get_parameter("curve_cooldown_s").as_double();
    curve_duration_s_       = get_parameter("curve_duration_s").as_double();

    reacquire_speed_         = get_parameter("reacquire_speed").as_double();
    reacquire_frames_needed_ = get_parameter("reacquire_frames_needed").as_int();
    reacquire_timeout_s_     = get_parameter("reacquire_timeout_s").as_double();

    enable_crossing_gate_         = get_parameter("enable_crossing_gate").as_bool();
    crossing_roi_y1_              = get_parameter("crossing_roi_y1").as_double();
    crossing_roi_y2_              = get_parameter("crossing_roi_y2").as_double();
    crossing_min_area_            = get_parameter("crossing_min_area").as_int();
    crossing_min_width_px_        = get_parameter("crossing_min_width_px").as_int();
    crossing_min_height_px_       = get_parameter("crossing_min_height_px").as_int();
    crossing_min_fill_ratio_      = get_parameter("crossing_min_fill_ratio").as_double();
    crossing_max_fill_ratio_      = get_parameter("crossing_max_fill_ratio").as_double();
    crossing_horizontal_ratio_    = get_parameter("crossing_horizontal_ratio").as_double();
    crossing_clear_frames_needed_ = get_parameter("crossing_clear_frames_needed").as_int();
    crossing_requires_three_lines_= get_parameter("crossing_requires_three_lines").as_bool();

    use_center_lane_only_ = get_parameter("use_center_lane_only").as_bool();
    center_lane_x_min_    = get_parameter("center_lane_x_min").as_double();
    center_lane_x_max_    = get_parameter("center_lane_x_max").as_double();

    enable_zebra_stop_roi_       = get_parameter("enable_zebra_stop_roi").as_bool();
    zebra_roi_x1_                = get_parameter("zebra_roi_x1").as_double();
    zebra_roi_x2_                = get_parameter("zebra_roi_x2").as_double();
    zebra_roi_y1_                = get_parameter("zebra_roi_y1").as_double();
    zebra_roi_y2_                = get_parameter("zebra_roi_y2").as_double();
    zebra_min_blobs_             = get_parameter("zebra_min_blobs").as_int();
    zebra_frames_needed_         = get_parameter("zebra_frames_needed").as_int();
    zebra_min_area_              = get_parameter("zebra_min_area").as_int();
    zebra_min_width_px_          = get_parameter("zebra_min_width_px").as_int();
    zebra_min_height_px_         = get_parameter("zebra_min_height_px").as_int();
    zebra_max_height_px_         = get_parameter("zebra_max_height_px").as_int();
    zebra_min_horizontal_ratio_  = get_parameter("zebra_min_horizontal_ratio").as_double();
    zebra_min_fill_ratio_        = get_parameter("zebra_min_fill_ratio").as_double();

    stop_at_zebra_enabled_     = get_parameter("stop_at_zebra_enabled").as_bool();
    give_away_stops_at_zebra_  = get_parameter("give_away_stops_at_zebra").as_bool();
    stop_zebra_arm_timeout_s_  = get_parameter("stop_zebra_arm_timeout_s").as_double();

    sign_timeout_s_         = get_parameter("sign_timeout_s").as_double();
    stop_duration_s_        = get_parameter("stop_duration_s").as_double();
    give_away_speed_factor_ = get_parameter("give_away_speed_factor").as_double();
    slow_speed_factor_      = get_parameter("slow_speed_factor").as_double();
    traffic_green_speed_multiplier_ = get_parameter("traffic_green_speed_multiplier").as_double();
    traffic_yellow_speed_multiplier_ = get_parameter("traffic_yellow_speed_multiplier").as_double();
    traffic_light_timeout_s_ = get_parameter("traffic_light_timeout_s").as_double();
    intersection_action_timeout_s_ = get_parameter("intersection_action_timeout_s").as_double();
    intersection_entry_speed_ = get_parameter("intersection_entry_speed").as_double();
    intersection_entry_duration_s_ = get_parameter("intersection_entry_duration_s").as_double();
    intersection_forward_duration_s_ = get_parameter("intersection_forward_duration_s").as_double();
    intersection_spin_speed_ = get_parameter("intersection_spin_speed").as_double();
    intersection_spin_duration_s_ = get_parameter("intersection_spin_duration_s").as_double();
    forward_duration_s_     = get_parameter("forward_duration_s").as_double();
  }

  // ===========================================================================
  // Camera
  // ===========================================================================
  void openCamera() {
    if (gst_) {
      std::string pipeline =
        "v4l2src device=" + device_ + " ! "
        "image/jpeg,width=" + std::to_string(width_) +
        ",height=" + std::to_string(height_) +
        ",framerate=" + std::to_string(fps_) + "/1 ! "
        "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true sync=false max-buffers=1";
      RCLCPP_INFO(get_logger(), "Opening GStreamer camera: %s", pipeline.c_str());
      cap_.open(pipeline, cv::CAP_GSTREAMER);
    } else {
      RCLCPP_INFO(get_logger(), "Opening camera source %d with V4L2.", source_);
      cap_.open(source_, cv::CAP_V4L2);
      cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
      cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
      cap_.set(cv::CAP_PROP_FPS,          fps_);
    }
  }

  // ===========================================================================
  // Callbacks
  // ===========================================================================
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    current_odom_x_ = msg->pose.pose.position.x;
    current_odom_y_ = msg->pose.pose.position.y;
    have_odom_ = true;
  }

  void signCallback(const std_msgs::msg::String::SharedPtr msg) {
    current_sign_   = normalizeSign(msg->data);
    last_sign_time_ = now();
    rememberSignIntent(current_sign_);
    RCLCPP_INFO(get_logger(), "Traffic sign detected: %s", current_sign_.c_str());
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv::Mat frame;
    if (!imageMsgToBgr(msg, frame)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Unsupported image encoding on %s: %s",
        image_topic_.c_str(), msg->encoding.c_str());
      return;
    }

    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = frame;
    have_latest_frame_ = true;
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================
  double clamp(double v, double lo, double hi) const {
    return std::max(lo, std::min(v, hi));
  }

  void publishWheels(double left, double right) {
    if (!enable_wheel_commands_) return;

    left  = clamp(left,  -max_speed_, max_speed_);
    right = clamp(right, -max_speed_, max_speed_);
    std_msgs::msg::Float32 l_msg, r_msg;
    l_msg.data = static_cast<float>(left  * left_motor_sign_);
    r_msg.data = static_cast<float>(right * right_motor_sign_);
    left_pub_->publish(l_msg);
    right_pub_->publish(r_msg);
  }

  void publishWheelStopDirect() {
    std_msgs::msg::Float32 msg; msg.data = 0.0f;
    if (left_pub_)  left_pub_->publish(msg);
    if (right_pub_) right_pub_->publish(msg);
  }

  void stopRobot() {
    if (!enable_wheel_commands_) return;

    publishWheelStopDirect();
  }

  void maybePublishDisabledStartupStop() {
    if (enable_wheel_commands_) return;
    if (disabled_startup_stop_remaining_ <= 0) return;

    publishWheelStopDirect();
    disabled_startup_stop_remaining_--;
  }

  bool imageMsgToBgr(const sensor_msgs::msg::Image::SharedPtr& msg, cv::Mat& frame) const {
    if (!msg || msg->data.empty() || msg->height == 0 || msg->width == 0) return false;

    const int h = static_cast<int>(msg->height);
    const int w = static_cast<int>(msg->width);
    const size_t step = static_cast<size_t>(msg->step);
    const uint8_t* data = msg->data.data();

    if (msg->encoding == "bgr8" || msg->encoding == "8UC3") {
      frame = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t*>(data), step).clone();
      return true;
    }
    if (msg->encoding == "rgb8") {
      cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t*>(data), step);
      cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
      return true;
    }
    if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
      cv::Mat gray(h, w, CV_8UC1, const_cast<uint8_t*>(data), step);
      cv::cvtColor(gray, frame, cv::COLOR_GRAY2BGR);
      return true;
    }
    if (msg->encoding == "bgra8") {
      cv::Mat bgra(h, w, CV_8UC4, const_cast<uint8_t*>(data), step);
      cv::cvtColor(bgra, frame, cv::COLOR_BGRA2BGR);
      return true;
    }
    if (msg->encoding == "rgba8") {
      cv::Mat rgba(h, w, CV_8UC4, const_cast<uint8_t*>(data), step);
      cv::cvtColor(rgba, frame, cv::COLOR_RGBA2BGR);
      return true;
    }

    return false;
  }

  bool getNextFrame(cv::Mat& frame) {
    if (use_image_topic_) {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!have_latest_frame_ || latest_frame_.empty()) return false;
      frame = latest_frame_.clone();
      return true;
    }

    return cap_.read(frame) && !frame.empty();
  }

  std::string normalizeSign(const std::string& raw) const {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char ch : raw) {
      if (ch == ' ' || ch == '-' || ch == '/') {
        out.push_back('_');
      } else {
        out.push_back(static_cast<char>(std::toupper(ch)));
      }
    }
    return out;
  }

  bool isTrafficGreen(const std::string& sign) const {
    return sign == "GREEN" ||
           sign == "GREEN_LIGHT" ||
           sign == "TRAFFIC_GREEN" ||
           sign == "TRAFFIC_LIGHT_GREEN";
  }

  bool isTrafficYellow(const std::string& sign) const {
    return sign == "YELLOW" ||
           sign == "YELLOW_LIGHT" ||
           sign == "AMBER" ||
           sign == "AMBER_LIGHT" ||
           sign == "TRAFFIC_YELLOW" ||
           sign == "TRAFFIC_LIGHT_YELLOW" ||
           sign == "TRAFFIC_AMBER" ||
           sign == "TRAFFIC_LIGHT_AMBER";
  }

  bool isTrafficRed(const std::string& sign) const {
    return sign == "RED" ||
           sign == "RED_LIGHT" ||
           sign == "TRAFFIC_RED" ||
           sign == "TRAFFIC_LIGHT_RED";
  }

  bool isTrafficLightSign(const std::string& sign) const {
    return isTrafficGreen(sign) || isTrafficYellow(sign) || isTrafficRed(sign);
  }

  bool isRouteLeft(const std::string& sign) const {
    return sign == "LEFT" || sign == "TURN_LEFT";
  }

  bool isRouteRight(const std::string& sign) const {
    return sign == "RIGHT" || sign == "TURN_RIGHT";
  }

  bool isRouteForward(const std::string& sign) const {
    return sign == "FORWARD" || sign == "STRAIGHT" || sign == "GO_STRAIGHT";
  }

  void rememberSignIntent(const std::string& sign) {
    const rclcpp::Time stamp = now();

    if (isTrafficGreen(sign)) {
      traffic_light_state_ = "GREEN";
      last_traffic_light_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Traffic light GREEN -> speed multiplier %.2f",
        traffic_green_speed_multiplier_);
    } else if (isTrafficYellow(sign)) {
      traffic_light_state_ = "YELLOW";
      last_traffic_light_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Traffic light YELLOW -> speed multiplier %.2f",
        traffic_yellow_speed_multiplier_);
    } else if (isTrafficRed(sign)) {
      traffic_light_state_ = "RED";
      last_traffic_light_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Traffic light RED armed. Will stop only at zebra/intersection.");
    }

    if (isRouteLeft(sign)) {
      pending_intersection_action_ = "LEFT";
      last_intersection_action_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Intersection action armed: LEFT at next zebra.");
    } else if (isRouteRight(sign)) {
      pending_intersection_action_ = "RIGHT";
      last_intersection_action_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Intersection action armed: RIGHT at next zebra.");
    } else if (isRouteForward(sign)) {
      pending_intersection_action_ = "FORWARD";
      last_intersection_action_time_ = stamp;
      RCLCPP_WARN(get_logger(), "Intersection action armed: FORWARD at next zebra.");
    }
  }

  double trafficSpeedMultiplierForMotion() const {
    if (traffic_light_state_ == "GREEN")
      return std::max(0.0, traffic_green_speed_multiplier_);
    if (traffic_light_state_ == "YELLOW")
      return std::max(0.0, traffic_yellow_speed_multiplier_);

    // RED is enforced by TRAFFIC_RED_WAIT at the zebra, not by stopping early.
    return 1.0;
  }

  void applyTrafficSpeedMultiplier(double& left, double& right) const {
    const double multiplier = trafficSpeedMultiplierForMotion();
    left *= multiplier;
    right *= multiplier;
  }

  // ==========================================================================
  // Camera calibration / lens flattening
  // ==========================================================================
  cv::Mat undistortFrame(const cv::Mat& frame) {
    if (!enable_undistort_) return frame;
    if (frame.empty()) return frame;

    const cv::Size frame_size(frame.cols, frame.rows);
    if (!undistort_maps_ready_ || frame_size != undistort_map_size_) {
      cv::Mat K = (cv::Mat_<double>(3,3) <<
        1036.5838636360986, 0.0, 603.55418660657051,
        0.0, 1032.8905876484835, 335.73449938027619,
        0.0, 0.0, 1.0);

      cv::Mat D = (cv::Mat_<double>(1,5) <<
        -0.35168816377968176,
         0.017337576601493002,
         0.0010445690989300657,
         0.0024864009424078954,
         0.097453194410303631);

      // If the calibration was made at a different resolution than the running
      // camera stream, scale fx/fy/cx/cy to the actual frame size.
      if (calib_width_ > 0 && calib_height_ > 0) {
        const double sx = static_cast<double>(frame.cols) / static_cast<double>(calib_width_);
        const double sy = static_cast<double>(frame.rows) / static_cast<double>(calib_height_);
        K.at<double>(0,0) *= sx;  // fx
        K.at<double>(0,2) *= sx;  // cx
        K.at<double>(1,1) *= sy;  // fy
        K.at<double>(1,2) *= sy;  // cy
      }

      cv::Mat newK = cv::getOptimalNewCameraMatrix(
        K, D, frame_size, undistort_alpha_, frame_size);

      cv::initUndistortRectifyMap(
        K, D, cv::Mat(), newK, frame_size, CV_16SC2,
        undistort_map1_, undistort_map2_);

      undistort_map_size_ = frame_size;
      undistort_maps_ready_ = true;

      RCLCPP_INFO(get_logger(),
        "Undistort maps ready: frame=%dx%d calib=%dx%d alpha=%.2f",
        frame.cols, frame.rows, calib_width_, calib_height_, undistort_alpha_);
    }

    cv::Mat flat;
    cv::remap(frame, flat, undistort_map1_, undistort_map2_, cv::INTER_LINEAR,
      cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return flat;
  }

  // ===========================================================================
  // Vision
  // ===========================================================================
cv::Mat buildTrackMask(int w, int h, std::vector<cv::Point>& poly) {
  // Tight vase-like ROI for the main center lane.
  // x: 0.0 left, 1.0 right
  // y: 0.0 top,  1.0 bottom
  const std::vector<cv::Point2f> norm = {
    {0.36f, 0.92f},  // bottom left
    {0.30f, 0.78f},  // lower left shoulder
    {0.44f, 0.54f},  // upper left neck
    {0.56f, 0.54f},  // upper right neck
    {0.70f, 0.78f},  // lower right shoulder
    {0.64f, 0.92f}   // bottom right
  };

  poly.clear();
  for (const auto& p : norm) {
    poly.emplace_back(
      static_cast<int>(p.x * static_cast<float>(w - 1)),
      static_cast<int>(p.y * static_cast<float>(h - 1))
    );
  }

  cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
  std::vector<std::vector<cv::Point>> polys = {poly};
  cv::fillPoly(mask, polys, 255);
  return mask;
}

cv::Mat cleanLineMask(const cv::Mat& raw_mask) {
  cv::Mat mask = raw_mask.clone();

  cv::Mat open_k = cv::Mat::ones(3, 3, CV_8UC1);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_k);

  cv::Mat close_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_k);

  return mask;
}

cv::Mat makeHSVLineMask(const cv::Mat& frame, const cv::Mat& valid_mask) {
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

  // Detect dark/black pixels.
  // Tune V max if needed: 80 strict, 100 normal, 120 loose.
  int v_max = std::min(black_ceiling_, 105);

  cv::Mat mask;
  cv::inRange(
    hsv,
    cv::Scalar(0, 0, 0),
    cv::Scalar(180, 180, v_max),
    mask
  );

  cv::bitwise_and(mask, valid_mask, mask);
  return cleanLineMask(mask);
}

cv::Mat makeOtsuLineMask(const cv::Mat& frame, const cv::Mat& valid_mask) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

  // Force outside ROI to white, so Otsu does not care about the full image.
  cv::Mat gray_roi = gray.clone();
  gray_roi.setTo(255, valid_mask == 0);

  cv::Mat otsu;
  cv::threshold(
    gray_roi,
    otsu,
    0,
    255,
    cv::THRESH_BINARY_INV + cv::THRESH_OTSU
  );

  // Extra black gate so tan road/white glare does not become line.
  cv::Mat black_gate;
  cv::inRange(gray, 0, black_ceiling_ + 10, black_gate);

  cv::Mat mask;
  cv::bitwise_and(otsu, black_gate, mask);
  cv::bitwise_and(mask, valid_mask, mask);

  return cleanLineMask(mask);
}

cv::Mat makeAdaptiveLineMask(const cv::Mat& frame, const cv::Mat& valid_mask) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

  int block = adaptive_block_;
  if (block % 2 == 0) block++;
  if (block < 3) block = 3;

  cv::Mat adaptive;
  cv::adaptiveThreshold(
    gray,
    adaptive,
    255,
    cv::ADAPTIVE_THRESH_GAUSSIAN_C,
    cv::THRESH_BINARY_INV,
    block,
    adaptive_c_
  );

  cv::Mat black_gate;
  cv::inRange(gray, 0, black_ceiling_, black_gate);

  cv::Mat mask;
  cv::bitwise_and(adaptive, black_gate, mask);
  cv::bitwise_and(mask, valid_mask, mask);

  return cleanLineMask(mask);
}

bool maskLooksUsable(
  const cv::Mat& mask,
  const cv::Mat& valid_mask,
  const std::vector<LaneLine>& lines
) const {
  if (lines.empty()) return false;

  int valid_px = cv::countNonZero(valid_mask);
  if (valid_px <= 0) return false;

  double ratio = static_cast<double>(cv::countNonZero(mask)) /
                 static_cast<double>(valid_px);

  // Too little = no useful line.
  // Too much = glare/seams/noise/intersection mess.
  if (ratio < 0.0015) return false;
  if (ratio > 0.30) return false;

  return true;
}

bool detectBestLineMask(
  const cv::Mat& frame,
  std::vector<cv::Point>& polygon,
  cv::Mat& selected_mask,
  std::vector<LaneLine>& selected_lines,
  int& selected_scan_y,
  int& selected_band_y1,
  int& selected_band_y2
) {
  const int h = frame.rows;
  const int w = frame.cols;

  cv::Mat valid_mask = buildTrackMask(w, h, polygon);

  cv::Mat hsv_mask      = makeHSVLineMask(frame, valid_mask);
  cv::Mat otsu_mask     = makeOtsuLineMask(frame, valid_mask);
  cv::Mat adaptive_mask = makeAdaptiveLineMask(frame, valid_mask);

  cv::Mat mixed_mask;
  cv::bitwise_or(hsv_mask, otsu_mask, mixed_mask);
  cv::bitwise_or(mixed_mask, adaptive_mask, mixed_mask);
  mixed_mask = cleanLineMask(mixed_mask);

  struct Candidate {
    std::string name;
    cv::Mat mask;
  };

  std::vector<Candidate> candidates = {
    {"HSV", hsv_mask},
    {"OTSU", otsu_mask},
    {"ADAPTIVE", adaptive_mask},
    {"MIXED", mixed_mask}
  };

  std::vector<LaneLine> best_lines;
  cv::Mat best_mask;
  std::string best_name = "NONE";
  int best_scan_y = 0;
  int best_band_y1 = 0;
  int best_band_y2 = 0;
  size_t best_count = 0;

  for (const auto& c : candidates) {
    int scan_y = 0;
    int band_y1 = 0;
    int band_y2 = 0;

    std::vector<LaneLine> lines =
      detectLaneLinesHoughPrimary(c.mask, scan_y, band_y1, band_y2);

    if (maskLooksUsable(c.mask, valid_mask, lines)) {
      selected_mask = c.mask;
      selected_lines = lines;
      selected_scan_y = scan_y;
      selected_band_y1 = band_y1;
      selected_band_y2 = band_y2;
      line_mask_method_ = c.name;
      return true;
    }

    if (lines.size() > best_count) {
      best_count = lines.size();
      best_lines = lines;
      best_mask = c.mask;
      best_name = c.name + "_WEAK";
      best_scan_y = scan_y;
      best_band_y1 = band_y1;
      best_band_y2 = band_y2;
    }
  }

  // Nothing passed the quality check. Use the least bad one.
  selected_mask = best_mask.empty() ? mixed_mask : best_mask;
  selected_lines = best_lines;
  selected_scan_y = best_scan_y;
  selected_band_y1 = best_band_y1;
  selected_band_y2 = best_band_y2;
  line_mask_method_ = best_name;

  return !selected_lines.empty();
}

  cv::Mat removeHorizontalCrossingBlobsForLane(const cv::Mat& mask) {
    // This removes wide horizontal zebra/crossing bars only for lane Hough detection.
    // The original mask is still used for crossing/zebra detection.
    cv::Mat lane_mask = mask.clone();

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& c : contours) {
      cv::Rect r = cv::boundingRect(c);

      int area = cv::countNonZero(mask(r));
      double fill = static_cast<double>(area) / std::max(1, r.width * r.height);
      double ratio = static_cast<double>(r.width) / std::max(1, r.height);

      bool looks_horizontal_bar =
        r.width >= 70 &&
        r.height >= 5 &&
        ratio >= 2.2 &&
        fill >= 0.20;

      bool in_front_area =
        r.y > static_cast<int>(mask.rows * 0.45);

      if (looks_horizontal_bar && in_front_area) {
        cv::rectangle(lane_mask, r, cv::Scalar(0), -1);
      }
    }

    return lane_mask;
  }

  bool detectCrossingBlob(const cv::Mat& line_mask, cv::Rect& best_rect, double& best_fill) {
    best_rect = cv::Rect(); best_fill = 0.0;
    if (!enable_crossing_gate_) { crossing_clear_count_++; return false; }
    const int h = line_mask.rows, w = line_mask.cols;
    int y1 = static_cast<int>(clamp(crossing_roi_y1_,0.0,1.0)*h);
    int y2 = static_cast<int>(clamp(crossing_roi_y2_,0.0,1.0)*h);
    y1 = std::max(0, std::min(y1, h-1));
    y2 = std::max(y1+1, std::min(y2, h));
    cv::Mat roi = line_mask(cv::Rect(0,y1,w,y2-y1)).clone();
    cv::Mat ck = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31,7));
    cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, ck);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    bool found = false; int best_area = 0;
    for (const auto& c : contours) {
      cv::Rect r  = cv::boundingRect(c);
      int  area   = cv::countNonZero(roi(r));
      double fill = static_cast<double>(area) / std::max(1, r.width*r.height);
      double ratio= static_cast<double>(r.width) / std::max(1, r.height);
      bool ok = area >= crossing_min_area_ && r.width >= crossing_min_width_px_ &&
                r.height >= crossing_min_height_px_ && fill >= crossing_min_fill_ratio_ &&
                fill <= crossing_max_fill_ratio_ && ratio >= crossing_horizontal_ratio_;
      if (ok && area > best_area) {
        best_area = area;
        best_rect = cv::Rect(r.x, r.y+y1, r.width, r.height);
        best_fill = fill; found = true;
      }
    }
    if (found) { crossing_clear_count_ = 0; crossing_active_ = true; }
    else {
      crossing_clear_count_++;
      if (crossing_clear_count_ >= crossing_clear_frames_needed_) crossing_active_ = false;
    }
    return crossing_active_;
  }

  bool shouldDetectCrossing(const std::vector<LaneLine>& lines, double error) {
    if (!enable_crossing_gate_) return false;

    // Do not look for crossings while the robot is already entering a curve.
    // In your screenshots the robot had a large left error, and crossing logic
    // was still involved in the decision. Crossings should only be checked when
    // the lane is stable and almost centered.
    if (std::abs(error) > curve_trigger_error_px_) return false;

    // A real crossing/intersection should show the three lane lines in this setup.
    // If we only see one/two lines, the robot is probably entering a curve or
    // partially losing the lane, so crossing blobs are disabled to prevent false stops.
    if (crossing_requires_three_lines_ && lines.size() < 3) return false;

    return true;
  }

  void clearCrossingState() {
    crossing_active_ = false;
    crossing_clear_count_ = crossing_clear_frames_needed_;
  }

  bool detectZebraCrossing(const cv::Mat& line_mask, cv::Rect& zebra_rect, int& zebra_count) {
    zebra_rect = cv::Rect();
    zebra_count = 0;
    if (!enable_zebra_stop_roi_) { zebra_active_ = false; zebra_seen_count_ = 0; return false; }

    const int h = line_mask.rows, w = line_mask.cols;
    int x1 = static_cast<int>(clamp(zebra_roi_x1_, 0.0, 1.0) * w);
    int x2 = static_cast<int>(clamp(zebra_roi_x2_, 0.0, 1.0) * w);
    int y1 = static_cast<int>(clamp(zebra_roi_y1_, 0.0, 1.0) * h);
    int y2 = static_cast<int>(clamp(zebra_roi_y2_, 0.0, 1.0) * h);
    x1 = std::max(0, std::min(x1, w-1));
    x2 = std::max(x1+1, std::min(x2, w));
    y1 = std::max(0, std::min(y1, h-1));
    y2 = std::max(y1+1, std::min(y2, h));

    cv::Mat roi = line_mask(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
    cv::Mat open_k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,3));
    cv::morphologyEx(roi, roi, cv::MORPH_OPEN, open_k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    bool have_union = false;
    cv::Rect union_rect;
    for (const auto& c : contours) {
      cv::Rect r = cv::boundingRect(c);
      int area_px = cv::countNonZero(roi(r));
      double ratio = static_cast<double>(r.width) / std::max(1, r.height);
      double fill  = static_cast<double>(area_px) / std::max(1, r.width*r.height);

      bool ok = area_px >= zebra_min_area_ &&
                r.width >= zebra_min_width_px_ &&
                r.height >= zebra_min_height_px_ &&
                r.height <= zebra_max_height_px_ &&
                ratio >= zebra_min_horizontal_ratio_ &&
                fill >= zebra_min_fill_ratio_;

      if (!ok) continue;
      cv::Rect abs_r(r.x + x1, r.y + y1, r.width, r.height);
      if (!have_union) { union_rect = abs_r; have_union = true; }
      else             { union_rect |= abs_r; }
      zebra_count++;
    }

    bool seen_now = zebra_count >= zebra_min_blobs_;
    if (seen_now) zebra_seen_count_++;
    else          zebra_seen_count_ = 0;

    zebra_active_ = zebra_seen_count_ >= zebra_frames_needed_;
    if (have_union) zebra_rect = union_rect;
    else            zebra_rect = cv::Rect(x1, y1, x2-x1, y2-y1);

    return zebra_active_;
  }

  std::vector<LaneLine> detectLaneLinesHoughPrimary(
    const cv::Mat& mask, int& scan_y, int& band_y1, int& band_y2)
  {
    const int h = mask.rows;
    const int w = mask.cols;

    scan_y  = static_cast<int>(h * scan_y_norm_);
    band_y1 = std::max(0, scan_y - band_height_ / 2);
    band_y2 = std::min(h, scan_y + band_height_ / 2);

    cv::Mat lane_only_mask = removeHorizontalCrossingBlobsForLane(mask);

    cv::Mat edges;
    cv::Canny(lane_only_mask, edges, 50, 150);

    std::vector<cv::Vec4i> hough_lines;
    cv::HoughLinesP(
      edges,
      hough_lines,
      1,
      CV_PI / 180.0,
      hough_threshold_,
      hough_min_length_,
      hough_max_gap_
    );

    struct LocalCandidate {
      LaneLine line;
      double x_at_scan;
      double length;
    };

    std::vector<LocalCandidate> candidates;

    for (const auto& l : hough_lines) {
      double x1 = static_cast<double>(l[0]);
      double y1 = static_cast<double>(l[1]);
      double x2 = static_cast<double>(l[2]);
      double y2 = static_cast<double>(l[3]);

      double dx = x2 - x1;
      double dy = y2 - y1;
      double length = std::sqrt(dx * dx + dy * dy);
      if (length < hough_min_length_) continue;

      // Reject mostly horizontal crossing/zebra bars.
      // angle_deg is 0/180 for horizontal, 90 for vertical.
      double angle_deg = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);
      double horizontal_angle = std::min(angle_deg, 180.0 - angle_deg);
      if (horizontal_angle < 12.0) continue;

      double x_at_scan = 0.0;
      if (std::abs(dy) > 1e-6) {
        double t = (static_cast<double>(scan_y) - y1) / dy;
        x_at_scan = x1 + t * dx;
      } else {
        x_at_scan = (x1 + x2) * 0.5;
      }

      if (x_at_scan < 0.0 || x_at_scan >= static_cast<double>(w)) continue;

      LaneLine ln;
      ln.x1 = static_cast<int>(x1);
      ln.y1 = static_cast<int>(y1);
      ln.x2 = static_cast<int>(x2);
      ln.y2 = static_cast<int>(y2);
      ln.cx = x_at_scan;

      candidates.push_back({ln, x_at_scan, length});
    }

    if (candidates.empty()) return {};

    std::sort(candidates.begin(), candidates.end(),
      [](const LocalCandidate& a, const LocalCandidate& b) {
        return a.x_at_scan < b.x_at_scan;
      });

    std::vector<std::vector<LocalCandidate>> grouped;
    std::vector<LocalCandidate> cur;
    cur.push_back(candidates[0]);

    for (size_t i = 1; i < candidates.size(); ++i) {
      double mx = 0.0;
      for (const auto& g : cur) mx += g.x_at_scan;
      mx /= static_cast<double>(cur.size());

      if (std::abs(candidates[i].x_at_scan - mx) <= group_gap_) {
        cur.push_back(candidates[i]);
      } else {
        grouped.push_back(cur);
        cur.clear();
        cur.push_back(candidates[i]);
      }
    }

    grouped.push_back(cur);

    std::vector<LaneLine> result;

    for (const auto& group : grouped) {
      double best_len = -1.0;
      LaneLine best_line;

      for (const auto& c : group) {
        if (c.length > best_len) {
          best_len = c.length;
          best_line = c.line;
        }
      }

      result.push_back(best_line);
    }

    std::sort(result.begin(), result.end(),
      [](const LaneLine& a, const LaneLine& b) {
        return a.cx < b.cx;
      });

    return result;
  }


  void chooseLaneTarget(
    const std::vector<LaneLine>& lines, int frame_width,
    double& target_x, double& error, std::string& tracking_mode)
  {
    double image_center = frame_width / 2.0;

    if (!use_center_lane_only_) {
      if (lines.size() >= 3) {
        target_x = lines[lines.size()/2].cx;
        tracking_mode = "MIDDLE_LINE";
      } else if (lines.size() == 2) {
        target_x = (lines[0].cx + lines[1].cx) / 2.0;
        tracking_mode = "TWO_LINES_MIDPOINT";
      } else if (lines.size() == 1) {
        double x = lines[0].cx;
        if (x < image_center) {
          target_x = x + fallback_offset_px_;
          tracking_mode = "ONE_LEFT_OFFSET";
        } else {
          target_x = x - fallback_offset_px_;
          tracking_mode = "ONE_RIGHT_OFFSET";
        }
      } else {
        target_x = image_center;
        tracking_mode = "LOST";
      }

      error = target_x - image_center;
      return;
    }

    double x_min = clamp(center_lane_x_min_, 0.0, 1.0) * frame_width;
    double x_max = clamp(center_lane_x_max_, 0.0, 1.0) * frame_width;
    if (x_min > x_max) std::swap(x_min, x_max);

    const LaneLine* best_center = nullptr;
    double best_dist = 1e9;

    for (const auto& ln : lines) {
      if (ln.cx < x_min || ln.cx > x_max) continue;

      double d = std::abs(ln.cx - image_center);
      if (d < best_dist) {
        best_dist = d;
        best_center = &ln;
      }
    }

    if (best_center) {
      target_x = best_center->cx;
      error = target_x - image_center;
      tracking_mode = "CENTER_LANE_ONLY";
    } else if (!lines.empty()) {
      double sum_x = 0.0;
      for (const auto& ln : lines) {
        sum_x += ln.cx;
      }

      double avg_x = sum_x / static_cast<double>(lines.size());
      target_x = avg_x;
      error = (avg_x - image_center) * 0.6;
      tracking_mode = "SIDE_LINES_STEER";
    } else {
      target_x = image_center;
      error = 0.0;
      tracking_mode = "CENTER_LANE_LOST";
    }
  }

  std::string turnDirectionFromSideLines(
    const std::vector<LaneLine>& lines, int frame_width,
    const std::string& fallback_direction) const
  {
    if (lines.empty()) return fallback_direction;
    double image_center = frame_width / 2.0;
    double sum_x = 0.0;
    for (const auto& ln : lines) sum_x += ln.cx;
    double avg_x = sum_x / static_cast<double>(lines.size());

    // If the remaining visible lane markers are mostly on the left side, the
    // road is bending left from the robot's point of view. Same for right.
    if (avg_x < image_center - deadband_px_) return "LEFT";
    if (avg_x > image_center + deadband_px_) return "RIGHT";
    return fallback_direction;
  }

  std::string directionFromError(double error) const {
    if (error >  deadband_px_) return "RIGHT";
    if (error < -deadband_px_) return "LEFT";
    return "CENTER";
  }

  // ---------------------------------------------------------------------------
  // fuzzyControl()
  //
  // CAMBIO: la corrección se clampea a ±(speed * 0.85) para garantizar que
  // ambas ruedas mantengan velocidad positiva (curvas suaves, no giro en eje).
  //
  // Antes: correction = kp * error  →  sin límite, podía generar rueda negativa.
  // Ahora: correction ≤ speed*0.85  →  rueda mínima = speed * 0.15 > 0 siempre.
  //
  // El error de visión NO se clampea aquí — llega completo desde chooseLaneTarget.
  // La saturación se aplica solo sobre la corrección, no sobre el error en sí.
  // ---------------------------------------------------------------------------
  void fuzzyControl(double error, double& left, double& right, std::string& control_mode) {
    double abs_error = std::abs(error);
    double kp, speed;

    if (abs_error < small_error_px_) {
      kp = kp_straight_; speed = speed_straight_; control_mode = "straight";
    } else if (abs_error < medium_error_px_) {
      kp = kp_soft_turn_; speed = speed_soft_turn_; control_mode = "soft_turn";
    } else {
      kp = kp_hard_turn_; speed = speed_hard_turn_; control_mode = "hard_turn";
    }

    // Conservative correction limit. This prevents the huge values visible in
    // your screenshot, such as L=-0.141 and R=1.338 during LINE_FOLLOW.
    const double limit = std::clamp(line_follow_correction_limit_, 0.10, 0.95);
    const double max_correction = speed * limit;
    double correction = std::clamp(turn_sign_ * kp * error, -max_correction, max_correction);

    left  = std::clamp(speed + correction, 0.0, max_speed_);
    right = std::clamp(speed - correction, 0.0, max_speed_);
  }

  // ===========================================================================
  // FSM: odometry curve
  //
  // CAMBIO: eliminado el dead reckoning (internal_curve_distance_m_).
  // La duración de la curva se controla por tiempo de pared (curve_duration_s_).
  // Esto elimina la dependencia de use_external_odom_ y variables de posición
  // interna, que era una estimación no confiable y difícil de calibrar.
  //
  // Si hay odometría externa (/odom), se usa la distancia real.
  // Si no la hay, se usa el tiempo transcurrido × curve_linear_speed_.
  // ===========================================================================
  void startOdomCurve(const std::string& direction, double radius_m, double angle_deg) {
    curve_direction_    = direction;
    curve_radius_m_cur_ = std::max(0.01, radius_m);

    // Distancia de arco objetivo (usada si hay /odom externo)
    curve_target_arc_m_ = std::abs(radius_m * angle_deg * CV_PI / 180.0);

    curve_start_x_  = current_odom_x_;
    curve_start_y_  = current_odom_y_;
    state_          = State::ODOM_CURVE;
    state_start_time_ = now();
    curve_trigger_count_ = 0;

    RCLCPP_WARN(get_logger(),
      "FSM -> ODOM_CURVE | dir=%s R=%.2f angle=%.1f arc=%.3fm t=%.2fs",
      direction.c_str(), radius_m, angle_deg,
      curve_target_arc_m_, curve_duration_s_);
  }

  std::string curveDirectionFromVision(
    const std::vector<LaneLine>& lines, double error,
    const std::string& fallback_direction) const
  {
    // Strong vision error wins. Negative error means the lane target is left,
    // so the robot must curve LEFT. Positive error means RIGHT.
    if (std::abs(error) >= curve_direction_override_error_px_) {
      return (error < 0.0) ? "LEFT" : "RIGHT";
    }

    // If there is still some weaker visual direction, use it before falling
    // back to the stored last direction.
    if (error < -deadband_px_) return "LEFT";
    if (error >  deadband_px_) return "RIGHT";

    if (fallback_direction == "LEFT" || fallback_direction == "RIGHT")
      return fallback_direction;
    return "RIGHT";
  }

  void maybeCorrectOdomCurveDirection(
    const std::vector<LaneLine>& lines, double error)
  {
    if (state_ != State::ODOM_CURVE) return;
    if (lines.empty()) return;
    if (std::abs(error) < curve_direction_override_error_px_) return;

    const std::string visual_dir = (error < 0.0) ? "LEFT" : "RIGHT";
    if (visual_dir != curve_direction_) {
      RCLCPP_WARN(get_logger(),
        "FSM: correcting ODOM_CURVE direction %s -> %s from vision error=%.1f",
        curve_direction_.c_str(), visual_dir.c_str(), error);
      curve_direction_ = visual_dir;
    }
  }

  void maybeStartOdomCurve(
    const std::vector<LaneLine>& /*lines*/, double /*error*/,
    bool /*crossing_now*/, const std::string& /*direction*/, bool /*center_lane_missing*/)
  {
    // ODOM CURVE DISABLED.
    // Keep normal lane following only.
    curve_trigger_count_ = 0;
    return;
  }

  void runOdomCurve(double& left, double& right) {
    const double R = curve_radius_m_cur_;
    const double b = wheel_base_m_;
    const double v = curve_linear_speed_;
    double inner = v * (R - b/2.0) / R;
    double outer = v * (R + b/2.0) / R;

    // Keep both wheels above the stall zone during a curve.
    // Without this, the inner wheel can be mathematically nonzero but too weak
    // to move the robot, making a turn look like a stop.
    const double min_curve = std::max(0.0, min_curve_wheel_speed_);
    if (inner > 0.0) inner = std::max(inner, min_curve);
    if (outer > 0.0) outer = std::max(outer, min_curve);

    if (curve_direction_ == "LEFT") { left = inner; right = outer; }
    else                            { left = outer; right = inner; }

    // Criterio de salida: tiempo de pared prioritario.
    // Si hay odometría externa, también se puede salir por distancia.
    bool done_by_time = (now() - state_start_time_).seconds() >= curve_duration_s_;
    bool done_by_odom = false;
    if (have_odom_) {
      double dx = current_odom_x_ - curve_start_x_;
      double dy = current_odom_y_ - curve_start_y_;
      done_by_odom = std::sqrt(dx*dx + dy*dy) >= curve_target_arc_m_;
    }

    if (done_by_time || done_by_odom) {
      state_ = State::REACQUIRE_LINE;
      state_start_time_ = now();
      reacquire_count_ = 0;
      RCLCPP_WARN(get_logger(),
        "FSM: ODOM_CURVE -> REACQUIRE_LINE | by=%s",
        done_by_time ? "time" : "odom");
    }
  }

  void runReacquireLine(
    const std::vector<LaneLine>& lines, bool crossing_now,
    double& left, double& right)
  {
    left = right = reacquire_speed_;
    if (!crossing_now && !lines.empty()) reacquire_count_++;
    else                                  reacquire_count_ = 0;

    bool enough  = reacquire_count_ >= reacquire_frames_needed_;
    bool timeout = (now() - state_start_time_).seconds() >= reacquire_timeout_s_;
    if (enough || timeout) {
      state_ = State::LINE_FOLLOW;
      last_curve_end_time_ = now();
      reacquire_count_ = 0;
      curve_trigger_count_ = 0;
      restoreMaxSpeed();
      RCLCPP_WARN(get_logger(),
        "FSM: REACQUIRE_LINE -> LINE_FOLLOW | frames=%d timeout=%d",
        reacquire_count_, timeout ? 1 : 0);
    }
  }

  // ===========================================================================
  // FSM: sign state handlers
  // ===========================================================================

  void runStopSign(double& left, double& right) {
    left = right = 0.0;
    if ((now() - state_start_time_).seconds() >= stop_duration_s_) {
      RCLCPP_WARN(get_logger(), "FSM: STOP_SIGN -> LINE_FOLLOW");
      current_sign_ = "NONE";
      pending_stop_at_zebra_ = false;
      pending_stop_reason_ = "NONE";
      state_ = State::LINE_FOLLOW;
    }
  }

  void runGiveAway(
    const std::vector<LaneLine>& lines, double error,
    bool crossing_now, double& left, double& right, std::string& control_mode)
  {
    if (!lines.empty()) {
      std::string cm;
      fuzzyControl(error, left, right, cm);
      left  *= give_away_speed_factor_;
      right *= give_away_speed_factor_;
      control_mode = "give_away_tracking";
    } else {
      // Sin línea: avanzar lento (mismo patrón que LINE_FOLLOW/LOST)
      left = right = speed_soft_turn_ * give_away_speed_factor_;
      control_mode = "give_away_lost";
    }
    if (current_sign_ == "NONE") {
      RCLCPP_WARN(get_logger(), "FSM: GIVE_AWAY -> LINE_FOLLOW (sign expired)");
      state_ = State::LINE_FOLLOW;
    }
    (void)crossing_now;
  }

  // LEFT_SIGN: dispara ODOM_CURVE a la izquierda directamente.
  void enterLeftSign() {
    RCLCPP_WARN(get_logger(), "FSM: LEFT_SIGN -> ODOM_CURVE LEFT");
    startOdomCurve("LEFT", curve_radius_m_, curve_angle_deg_);
    current_sign_ = "NONE";
  }

  // RIGHT_SIGN: dispara ODOM_CURVE a la derecha directamente.
  void enterRightSign() {
    RCLCPP_WARN(get_logger(), "FSM: RIGHT_SIGN -> ODOM_CURVE RIGHT");
    startOdomCurve("RIGHT", curve_radius_m_, curve_angle_deg_);
    current_sign_ = "NONE";
  }

  // ---------------------------------------------------------------------------
  // FORWARD_SIGN
  //
  // CAMBIO: ya NO usa startOdomCurve() ni last_non_center_direction_.
  // El robot avanza recto (left == right) durante forward_duration_s_ segundos.
  // Esto elimina el problema donde FORWARD heredaba una dirección lateral
  // y producía una curva inesperada en lugar de ir recto.
  // ---------------------------------------------------------------------------
  void runForwardSign(double& left, double& right) {
    left = right = speed_straight_;   // ambas ruedas iguales → recto
    if ((now() - state_start_time_).seconds() >= forward_duration_s_) {
      RCLCPP_WARN(get_logger(), "FSM: FORWARD_SIGN -> LINE_FOLLOW");
      current_sign_ = "NONE";
      state_ = State::LINE_FOLLOW;
    }
  }

  void enterTrafficRedWait() {
    state_ = State::TRAFFIC_RED_WAIT;
    state_start_time_ = now();
    current_sign_ = "NONE";
    RCLCPP_WARN(get_logger(),
      "FSM: LINE_FOLLOW -> TRAFFIC_RED_WAIT | action_after_green=%s",
      pending_intersection_action_.c_str());
  }

  void maybeStartIntersectionAction() {
    startIntersectionAction();
  }

  void startIntersectionAction() {
    active_intersection_action_ = pending_intersection_action_;
    if (active_intersection_action_ != "LEFT" &&
        active_intersection_action_ != "RIGHT" &&
        active_intersection_action_ != "FORWARD") {
      active_intersection_action_ = "FORWARD";
    }

    pending_intersection_action_ = "NONE";
    state_ = State::INTERSECTION_FORWARD;
    state_start_time_ = now();

    RCLCPP_WARN(get_logger(),
      "FSM: intersection action start | action=%s entry=%.2fs spin=%.2fs",
      active_intersection_action_.c_str(),
      intersection_entry_duration_s_, intersection_spin_duration_s_);
  }

  void finishIntersectionAction(bool reacquire) {
    active_intersection_action_ = "NONE";
    zebra_seen_count_ = 0;
    zebra_active_ = false;
    if (reacquire) {
      state_ = State::REACQUIRE_LINE;
      state_start_time_ = now();
      reacquire_count_ = 0;
      RCLCPP_WARN(get_logger(), "FSM: INTERSECTION_SPIN -> REACQUIRE_LINE");
    } else {
      state_ = State::LINE_FOLLOW;
      state_start_time_ = now();
      RCLCPP_WARN(get_logger(), "FSM: INTERSECTION_FORWARD -> LINE_FOLLOW");
    }
  }

  void runTrafficRedWait(double& left, double& right) {
    left = right = 0.0;
    if (traffic_light_state_ != "RED") {
      maybeStartIntersectionAction();
      RCLCPP_WARN(get_logger(), "FSM: TRAFFIC_RED_WAIT released by traffic=%s",
        traffic_light_state_.c_str());
    }
  }

  void runIntersectionForward(double& left, double& right) {
    left = right = intersection_entry_speed_;

    const bool turning =
      active_intersection_action_ == "LEFT" ||
      active_intersection_action_ == "RIGHT";
    const double duration = turning ?
      intersection_entry_duration_s_ : intersection_forward_duration_s_;

    if ((now() - state_start_time_).seconds() < duration) return;

    if (turning) {
      state_ = State::INTERSECTION_SPIN;
      state_start_time_ = now();
      RCLCPP_WARN(get_logger(), "FSM: INTERSECTION_FORWARD -> INTERSECTION_SPIN %s",
        active_intersection_action_.c_str());
    } else {
      finishIntersectionAction(false);
    }
  }

  void runIntersectionSpin(double& left, double& right) {
    const double spin = std::max(0.0, intersection_spin_speed_);
    if (active_intersection_action_ == "LEFT") {
      left = -spin;
      right = spin;
    } else {
      left = spin;
      right = -spin;
    }

    if ((now() - state_start_time_).seconds() >= intersection_spin_duration_s_) {
      finishIntersectionAction(true);
    }
  }

  void enterSlowMode(const std::string& reason) {
    if (!slow_mode_active_) {
      slow_mode_active_ = true;
      max_speed_ = normal_max_speed_ * slow_speed_factor_;
      RCLCPP_WARN(get_logger(), "FSM: %s -> max_speed=%.3f (factor=%.2f)",
        reason.c_str(), max_speed_, slow_speed_factor_);
    }
  }

  void restoreMaxSpeed() {
    if (slow_mode_active_) {
      slow_mode_active_ = false;
      max_speed_ = normal_max_speed_;
      RCLCPP_WARN(get_logger(), "FSM: max_speed restored to %.3f", max_speed_);
    }
  }

  // ---------------------------------------------------------------------------
  // runSlowMode()
  //
  // CAMBIO: la reducción de velocidad se aplica multiplicando las velocidades
  // calculadas por slow_speed_factor_, en lugar de depender del clamp de
  // max_speed_. Esto garantiza que el robot se mueva físicamente más lento
  // incluso si las velocidades base son menores que max_speed_.
  // ---------------------------------------------------------------------------
  void runSlowMode(
    const std::vector<LaneLine>& lines, double error,
    bool crossing_now, double& left, double& right,
    std::string& control_mode, const std::string& mode_name)
  {
    if (!lines.empty()) {
      std::string cm;
      fuzzyControl(error, left, right, cm);
    } else {
      left = right = speed_soft_turn_;
    }
    // Aplicar reducción directamente sobre las velocidades calculadas
    left  *= slow_speed_factor_;
    right *= slow_speed_factor_;
    control_mode = mode_name;

    if (current_sign_ == "NONE") {
      restoreMaxSpeed();
      RCLCPP_WARN(get_logger(), "FSM: %s -> LINE_FOLLOW (sign expired)", mode_name.c_str());
      state_ = State::LINE_FOLLOW;
    }
    (void)crossing_now;
  }

  // ===========================================================================
  // Sign timeout & transition
  // ===========================================================================
  void checkSignTimeout() {
    const rclcpp::Time stamp = now();

    if (last_sign_time_.nanoseconds() != 0 &&
        current_sign_ != "NONE" &&
        (stamp - last_sign_time_).seconds() > sign_timeout_s_) {
      RCLCPP_INFO(get_logger(), "Sign timeout: %s -> NONE", current_sign_.c_str());
      current_sign_ = "NONE";
    }

    if (last_traffic_light_time_.nanoseconds() != 0 &&
        traffic_light_state_ != "NONE" &&
        (stamp - last_traffic_light_time_).seconds() > traffic_light_timeout_s_) {
      RCLCPP_INFO(get_logger(), "Traffic light timeout: %s -> NONE", traffic_light_state_.c_str());
      traffic_light_state_ = "NONE";
    }

    if (last_intersection_action_time_.nanoseconds() != 0 &&
        pending_intersection_action_ != "NONE" &&
        (stamp - last_intersection_action_time_).seconds() > intersection_action_timeout_s_) {
      RCLCPP_INFO(get_logger(), "Intersection action timeout: %s -> NONE",
        pending_intersection_action_.c_str());
      pending_intersection_action_ = "NONE";
    }
  }

  bool isStopRequiredSign(const std::string& sign) const {
    return sign == "STOP" ||
           (give_away_stops_at_zebra_ && sign == "GIVE_AWAY");
  }

  void enterStopSignNow(const std::string& reason) {
    state_ = State::STOP_SIGN;
    state_start_time_ = now();
    current_sign_ = "NONE";
    pending_stop_at_zebra_ = false;
    pending_stop_reason_ = "NONE";
    RCLCPP_WARN(get_logger(), "FSM: LINE_FOLLOW -> STOP_SIGN | reason=%s", reason.c_str());
  }

  void maybeEnterSignState(bool zebra_now) {
    if (state_ != State::LINE_FOLLOW) return;

    if (zebra_now && traffic_light_state_ == "RED") {
      enterTrafficRedWait();
      return;
    }

    if (zebra_now && pending_intersection_action_ != "NONE") {
      maybeStartIntersectionAction();
      current_sign_ = "NONE";
      return;
    }

    if (isTrafficLightSign(current_sign_)) {
      if (zebra_now && isTrafficRed(current_sign_)) {
        enterTrafficRedWait();
      }
      return;
    }

    // If a stop-required sign was seen earlier, wait until the zebra/stop-line
    // ROI is reached. This avoids stopping immediately at the sign instead of
    // at the intended stop position.
    if (pending_stop_at_zebra_) {
      if (zebra_now) {
        enterStopSignNow("zebra_" + pending_stop_reason_);
        return;
      }
      if ((now() - pending_stop_start_time_).seconds() > stop_zebra_arm_timeout_s_) {
        RCLCPP_WARN(get_logger(), "STOP-at-zebra expired without zebra detection | reason=%s", pending_stop_reason_.c_str());
        pending_stop_at_zebra_ = false;
        pending_stop_reason_ = "NONE";
      }
    }

    if (isStopRequiredSign(current_sign_)) {
      if (stop_at_zebra_enabled_) {
        pending_stop_at_zebra_ = true;
        pending_stop_reason_ = current_sign_;
        pending_stop_start_time_ = now();
        current_sign_ = "NONE";
        RCLCPP_WARN(get_logger(), "STOP sign armed. Waiting for zebra ROI.");
        if (zebra_now) {
          enterStopSignNow("zebra_" + pending_stop_reason_);
        }
      } else {
        enterStopSignNow(current_sign_);
      }
    } else if (isRouteLeft(current_sign_) ||
               isRouteRight(current_sign_) ||
               isRouteForward(current_sign_)) {
      current_sign_ = "NONE";
      if (zebra_now) {
        maybeStartIntersectionAction();
      }
    } else if (current_sign_ == "GIVE_AWAY") {
      state_ = State::GIVE_AWAY;
      state_start_time_ = now();
      RCLCPP_WARN(get_logger(), "FSM: LINE_FOLLOW -> GIVE_AWAY");
    } else if (current_sign_ == "TURN_LEFT") {
      state_ = State::LEFT_SIGN;
      enterLeftSign();
    } else if (current_sign_ == "TURN_RIGHT") {
      state_ = State::RIGHT_SIGN;
      enterRightSign();
    } else if (current_sign_ == "FORWARD") {
      state_ = State::FORWARD_SIGN;
      state_start_time_ = now();
      current_sign_ = "NONE";
      RCLCPP_WARN(get_logger(), "FSM: LINE_FOLLOW -> FORWARD_SIGN (%.1fs recto)", forward_duration_s_);
    } else if (current_sign_ == "ROADWORK") {
      state_ = State::ROADWORK;
      state_start_time_ = now();
      enterSlowMode("ROADWORK");
    } else if (current_sign_ == "WORKERS_AHEAD") {
      state_ = State::WORKERS_AHEAD;
      state_start_time_ = now();
      enterSlowMode("WORKERS_AHEAD");
    }
  }

  // ===========================================================================
  // State name
  // ===========================================================================
  std::string stateName() const {
    switch (state_) {
      case State::LINE_FOLLOW:    return "LINE_FOLLOW";
      case State::ODOM_CURVE:     return "ODOM_CURVE";
      case State::REACQUIRE_LINE: return "REACQUIRE_LINE";
      case State::STOP_SIGN:      return "STOP_SIGN";
      case State::GIVE_AWAY:      return "GIVE_AWAY";
      case State::LEFT_SIGN:      return "LEFT_SIGN";
      case State::RIGHT_SIGN:     return "RIGHT_SIGN";
      case State::FORWARD_SIGN:   return "FORWARD_SIGN";
      case State::TRAFFIC_RED_WAIT: return "TRAFFIC_RED_WAIT";
      case State::INTERSECTION_FORWARD: return "INTERSECTION_FORWARD";
      case State::INTERSECTION_SPIN: return "INTERSECTION_SPIN";
      case State::ROADWORK:       return "ROADWORK";
      case State::WORKERS_AHEAD:  return "WORKERS_AHEAD";
    }
    return "UNKNOWN";
  }

  // ===========================================================================
  // Debug overlay
  // ===========================================================================
  cv::Mat makeDebugFrame(
    const cv::Mat& frame, const cv::Mat& mask,
    const std::vector<cv::Point>& polygon,
    const std::vector<LaneLine>& lines,
    double target_x, double error,
    const std::string& tracking_mode, const std::string& direction,
    int scan_y, int band_y1, int band_y2,
    double left_speed, double right_speed,
    bool crossing_now, const cv::Rect& crossing_rect, double crossing_fill,
    bool zebra_now, const cv::Rect& zebra_rect, int zebra_count)
  {
    cv::Mat debug = frame.clone();
    const int h = debug.rows, w = debug.cols;
    std::vector<std::vector<cv::Point>> polys = {polygon};
    cv::polylines(debug, polys, true, cv::Scalar(0,255,255), 2);
    cv::rectangle(debug, cv::Point(0,band_y1), cv::Point(w-1,band_y2), cv::Scalar(255,0,255), 1);
    cv::line(debug, cv::Point(w/2,band_y1), cv::Point(w/2,band_y2), cv::Scalar(0,0,255), 2);
    cv::line(debug, cv::Point(static_cast<int>(target_x),band_y1),
                    cv::Point(static_cast<int>(target_x),band_y2), cv::Scalar(255,0,0), 2);

    for (size_t i = 0; i < lines.size(); ++i) {
      const auto& ln = lines[i];
      cv::rectangle(debug, cv::Rect(ln.x,ln.y,ln.w,ln.h), cv::Scalar(0,255,0), 2);
      cv::circle(debug, cv::Point(static_cast<int>(ln.cx),scan_y), 6, cv::Scalar(0,255,0), -1);
      cv::line(debug, cv::Point(ln.x1,ln.y1), cv::Point(ln.x2,ln.y2), cv::Scalar(255,0,255), 2);
      cv::putText(debug,
        "L"+std::to_string(i+1)+" H="+std::to_string(static_cast<int>(ln.angle)),
        cv::Point(ln.x, std::max(ln.y-5,20)),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,0), 2);
    }

    if (crossing_now) {
      cv::rectangle(debug, crossing_rect, cv::Scalar(0,165,255), 3);
      cv::putText(debug, "CROSSING BLOB",
        cv::Point(crossing_rect.x, std::max(crossing_rect.y-8,20)),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,165,255), 2);
    }

    // Zebra ROI / stop-line region. Blue rectangle = ROI; red rectangle = detected zebra bars.
    int zx1 = static_cast<int>(clamp(zebra_roi_x1_,0.0,1.0)*w);
    int zx2 = static_cast<int>(clamp(zebra_roi_x2_,0.0,1.0)*w);
    int zy1 = static_cast<int>(clamp(zebra_roi_y1_,0.0,1.0)*h);
    int zy2 = static_cast<int>(clamp(zebra_roi_y2_,0.0,1.0)*h);
    cv::rectangle(debug, cv::Rect(zx1,zy1,std::max(1,zx2-zx1),std::max(1,zy2-zy1)), cv::Scalar(255,0,0), 2);
    if (zebra_now) {
      cv::rectangle(debug, zebra_rect, cv::Scalar(0,0,255), 3);
      cv::putText(debug, "ZEBRA STOP",
        cv::Point(zebra_rect.x, std::max(zebra_rect.y-8,20)),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
    }

    cv::Scalar dir_color(0,255,0);
    if (direction=="LEFT")  dir_color=cv::Scalar(0,255,255);
    if (direction=="RIGHT") dir_color=cv::Scalar(255,255,0);

    cv::Scalar sign_color(200,200,200);
    if (current_sign_=="STOP")          sign_color=cv::Scalar(0,0,255);
    if (isTrafficRed(current_sign_))    sign_color=cv::Scalar(0,0,255);
    if (isTrafficYellow(current_sign_)) sign_color=cv::Scalar(0,255,255);
    if (isTrafficGreen(current_sign_))  sign_color=cv::Scalar(0,255,0);
    if (current_sign_=="GIVE_AWAY")     sign_color=cv::Scalar(0,165,255);
    if (current_sign_=="TURN_LEFT")     sign_color=cv::Scalar(0,255,255);
    if (current_sign_=="TURN_RIGHT")    sign_color=cv::Scalar(255,255,0);
    if (current_sign_=="FORWARD")       sign_color=cv::Scalar(0,255,0);
    if (current_sign_=="ROADWORK")      sign_color=cv::Scalar(0,128,255);
    if (current_sign_=="WORKERS_AHEAD") sign_color=cv::Scalar(0,128,255);

    std::string speed_label = slow_mode_active_ ?
      ("SLOW x"+std::to_string(slow_speed_factor_).substr(0,3)) : "NORMAL";
    std::string traffic_label = traffic_light_state_;
    if (traffic_light_state_ == "GREEN") {
      traffic_label += "x" + std::to_string(traffic_green_speed_multiplier_).substr(0,3);
    } else if (traffic_light_state_ == "YELLOW") {
      traffic_label += "x" + std::to_string(traffic_yellow_speed_multiplier_).substr(0,3);
    } else if (traffic_light_state_ == "RED") {
      traffic_label += "x0@ZEBRA";
    }
    std::string wheel_command_label = enable_wheel_commands_ ? "cmd=ON" : "cmd=OFF";

    cv::rectangle(debug, cv::Point(0,0), cv::Point(w,225), cv::Scalar(0,0,0), -1);
    cv::putText(debug,
      "CENTER_ZEBRA_UNDISTORT_V4 | FSM: "+stateName()+" | LANE: "+direction,
      cv::Point(20,35), cv::FONT_HERSHEY_SIMPLEX, 0.8, dir_color, 2, cv::LINE_AA);
    cv::putText(debug,
      "error="+std::to_string(static_cast<int>(error))+
      "px | lines="+std::to_string(lines.size())+
      " | "+tracking_mode+" | cross="+std::to_string(crossing_now?1:0)+
      " | zebra="+std::to_string(zebra_now?1:0),
      cv::Point(20,70), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 2, cv::LINE_AA);
    cv::putText(debug,
      "L="+std::to_string(left_speed).substr(0,5)+
      " R="+std::to_string(right_speed).substr(0,5)+
      " | spd="+speed_label+" | tl="+traffic_label+" | "+wheel_command_label,
      cv::Point(20,105), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 2, cv::LINE_AA);
    cv::putText(debug,
      "curve_trigger>"+std::to_string(static_cast<int>(curve_trigger_error_px_))+
      "px | dur="+std::to_string(curve_duration_s_).substr(0,4)+"s",
      cv::Point(20,140), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 2, cv::LINE_AA);
    cv::putText(debug,
      "cross_fill="+std::to_string(crossing_fill).substr(0,4)+
      " | zebra_blobs="+std::to_string(zebra_count)+
      " | undist="+std::to_string(enable_undistort_?1:0)+
      " | action="+pending_intersection_action_,
      cv::Point(20,175), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255,255,255), 2, cv::LINE_AA);
    cv::putText(debug,
      "SIGN: "+current_sign_+" | active="+active_intersection_action_,
      cv::Point(20,210), cv::FONT_HERSHEY_SIMPLEX, 0.65, sign_color, 2, cv::LINE_AA);

    if (side_by_side_) {
      cv::Mat mask_bgr;
      cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
      cv::polylines(mask_bgr, polys, true, cv::Scalar(0,255,255), 2);
      cv::rectangle(mask_bgr, cv::Point(0,band_y1), cv::Point(w-1,band_y2), cv::Scalar(255,0,255), 1);
      if (crossing_now) cv::rectangle(mask_bgr, crossing_rect, cv::Scalar(0,165,255), 3);
      cv::rectangle(mask_bgr, cv::Rect(zx1,zy1,std::max(1,zx2-zx1),std::max(1,zy2-zy1)), cv::Scalar(255,0,0), 2);
      if (zebra_now) cv::rectangle(mask_bgr, zebra_rect, cv::Scalar(0,0,255), 3);

      cv::rectangle(mask_bgr, cv::Point(0,0), cv::Point(w,132), cv::Scalar(0,0,0), -1);
      cv::putText(mask_bgr,
        "FSM: "+stateName()+" | "+tracking_mode,
        cv::Point(12,26), cv::FONT_HERSHEY_SIMPLEX, 0.52, dir_color, 2, cv::LINE_AA);
      cv::putText(mask_bgr,
        "err="+std::to_string(static_cast<int>(error))+
        " lines="+std::to_string(lines.size())+
        " cross="+std::to_string(crossing_now?1:0)+
        " zebra="+std::to_string(zebra_now?1:0)+
        " zcnt="+std::to_string(zebra_count),
        cv::Point(12,52), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(255,255,255), 1, cv::LINE_AA);
      cv::putText(mask_bgr,
        "L="+std::to_string(left_speed).substr(0,5)+
        " R="+std::to_string(right_speed).substr(0,5)+
        " tl="+traffic_label+
        " sign="+current_sign_+
        " "+wheel_command_label,
        cv::Point(12,78), cv::FONT_HERSHEY_SIMPLEX, 0.43, sign_color, 1, cv::LINE_AA);
      cv::putText(mask_bgr,
        "action="+pending_intersection_action_+
        " active="+active_intersection_action_,
        cv::Point(12,104), cv::FONT_HERSHEY_SIMPLEX, 0.43, cv::Scalar(255,255,255), 1, cv::LINE_AA);

      cv::Mat camera_view = frame.clone();
      cv::hconcat(camera_view, mask_bgr, debug);
    }
    return debug;
  }

  // ===========================================================================
  // Video writer
  // ===========================================================================
  void writeVideo(const cv::Mat& frame) {
    if (video_failed_) return;
    if (!writer_.isOpened()) {
      writer_.open(out_path_,
        cv::VideoWriter::fourcc('M','J','P','G'),
        video_fps_, cv::Size(frame.cols, frame.rows), true);
      if (!writer_.isOpened()) {
        RCLCPP_ERROR(get_logger(), "Could not open video writer: %s", out_path_.c_str());
        video_failed_ = true; return;
      }
      RCLCPP_INFO(get_logger(), "Video writer opened: %dx%d @ %.1f FPS",
        frame.cols, frame.rows, video_fps_);
    }
    writer_.write(frame);
  }

  // ===========================================================================
  // Main control loop
  // ===========================================================================
  void controlStep() {
    if (seconds_ > 0.0 && now() >= end_time_) {
      RCLCPP_INFO(get_logger(), "Time limit reached. Stopping.");
      stopRobot(); rclcpp::shutdown(); return;
    }

    checkSignTimeout();
    maybePublishDisabledStartupStop();

    cv::Mat frame;
    if (!getNextFrame(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "No camera frame.");
      stopRobot();
      return;
    }

    // Flatten lens distortion first, so line/zebra geometry is measured on the corrected image.
    frame = undistortFrame(frame);

    // Vision
  std::vector<cv::Point> polygon;
  cv::Mat mask;
  int scan_y = 0;
  int band_y1 = 0;
  int band_y2 = 0;
  std::vector<LaneLine> lines;

  detectBestLineMask(
    frame,
    polygon,
    mask,
    lines,
    scan_y,
    band_y1,
    band_y2
  );
double target_x = frame.cols/2.0, error = 0.0;
    std::string tracking_mode = "LOST";
    chooseLaneTarget(lines, frame.cols, target_x, error, tracking_mode);
    std::string direction = directionFromError(error);

    const bool center_lane_missing =
      use_center_lane_only_ && (tracking_mode == "CENTER_LANE_LOST");

    if (tracking_mode == "SIDE_LINES_STEER" || tracking_mode == "CENTER_LANE_LOST") {
      // Actualizar dirección desde laterales para que ODOM_CURVE tome la correcta
      direction = turnDirectionFromSideLines(lines, frame.cols, last_non_center_direction_);
    }
    if (direction == "LEFT" || direction == "RIGHT") last_non_center_direction_ = direction;

    cv::Rect crossing_rect; double crossing_fill=0.0;
    bool crossing_now = false;
    if (shouldDetectCrossing(lines, error)) {
      crossing_now = detectCrossingBlob(mask, crossing_rect, crossing_fill);
    } else {
      clearCrossingState();
    }

    cv::Rect zebra_rect; int zebra_count=0;
    bool zebra_now = detectZebraCrossing(mask, zebra_rect, zebra_count);

    double left_speed=0.0, right_speed=0.0;
    std::string control_mode = "none";

    switch (state_) {

      case State::LINE_FOLLOW:
        maybeEnterSignState(zebra_now);
        if (state_ == State::LINE_FOLLOW) {
          maybeStartOdomCurve(lines, error, crossing_now, direction, center_lane_missing);
          if (state_ == State::ODOM_CURVE) {
            maybeCorrectOdomCurveDirection(lines, error);
            runOdomCurve(left_speed, right_speed);
            tracking_mode = "ODOM_CURVE_"+curve_direction_;
          } else if (!lines.empty()) {
            lost_counter_ = 0;
            fuzzyControl(error, left_speed, right_speed, control_mode);
          } else {
            lost_counter_++;
            if (lost_counter_ < max_lost_frames_) {
              tracking_mode = "LOST_FORWARD";
              left_speed = right_speed = speed_soft_turn_;
            } else {
              tracking_mode = "SEARCH";
              left_speed  = -search_speed_;
              right_speed =  search_speed_;
            }
          }
        } else {
          // Señal cambió el estado — ejecutar en este mismo paso
          goto run_new_state;
        }
        break;

      case State::ODOM_CURVE:
        maybeCorrectOdomCurveDirection(lines, error);
        runOdomCurve(left_speed, right_speed);
        tracking_mode = "ODOM_CURVE_"+curve_direction_;
        break;

      case State::REACQUIRE_LINE:
        runReacquireLine(lines, crossing_now, left_speed, right_speed);
        tracking_mode = "REACQUIRE_LINE";
        break;

      case State::STOP_SIGN:
        runStopSign(left_speed, right_speed);
        tracking_mode = "STOP_SIGN";
        break;

      case State::GIVE_AWAY:
        runGiveAway(lines, error, crossing_now, left_speed, right_speed, control_mode);
        tracking_mode = "GIVE_AWAY";
        break;

      case State::LEFT_SIGN:
        enterLeftSign();   // Transición inmediata a ODOM_CURVE
        runOdomCurve(left_speed, right_speed);
        tracking_mode = "LEFT_SIGN";
        break;

      case State::RIGHT_SIGN:
        enterRightSign();  // Transición inmediata a ODOM_CURVE
        runOdomCurve(left_speed, right_speed);
        tracking_mode = "RIGHT_SIGN";
        break;

      case State::FORWARD_SIGN:
        runForwardSign(left_speed, right_speed);
        tracking_mode = "FORWARD_SIGN";
        break;

      case State::TRAFFIC_RED_WAIT:
        runTrafficRedWait(left_speed, right_speed);
        tracking_mode = "TRAFFIC_RED_WAIT";
        break;

      case State::INTERSECTION_FORWARD:
        runIntersectionForward(left_speed, right_speed);
        tracking_mode = "INTERSECTION_FORWARD_"+active_intersection_action_;
        break;

      case State::INTERSECTION_SPIN:
        runIntersectionSpin(left_speed, right_speed);
        tracking_mode = "INTERSECTION_SPIN_"+active_intersection_action_;
        break;

      case State::ROADWORK:
        runSlowMode(lines,error,crossing_now,left_speed,right_speed,control_mode,"ROADWORK");
        tracking_mode = "ROADWORK";
        break;

      case State::WORKERS_AHEAD:
        runSlowMode(lines,error,crossing_now,left_speed,right_speed,control_mode,"WORKERS_AHEAD");
        tracking_mode = "WORKERS_AHEAD";
        break;

      run_new_state:
      default:
        switch (state_) {
          case State::STOP_SIGN:
            runStopSign(left_speed,right_speed); tracking_mode="STOP_SIGN"; break;
          case State::GIVE_AWAY:
            runGiveAway(lines,error,crossing_now,left_speed,right_speed,control_mode);
            tracking_mode="GIVE_AWAY"; break;
          case State::FORWARD_SIGN:
            runForwardSign(left_speed,right_speed); tracking_mode="FORWARD_SIGN"; break;
          case State::TRAFFIC_RED_WAIT:
            runTrafficRedWait(left_speed,right_speed); tracking_mode="TRAFFIC_RED_WAIT"; break;
          case State::INTERSECTION_FORWARD:
            runIntersectionForward(left_speed,right_speed);
            tracking_mode="INTERSECTION_FORWARD_"+active_intersection_action_; break;
          case State::INTERSECTION_SPIN:
            runIntersectionSpin(left_speed,right_speed);
            tracking_mode="INTERSECTION_SPIN_"+active_intersection_action_; break;
          case State::ODOM_CURVE:
            maybeCorrectOdomCurveDirection(lines, error);
            runOdomCurve(left_speed,right_speed);
            tracking_mode="ODOM_CURVE_"+curve_direction_; break;
          case State::ROADWORK:
            runSlowMode(lines,error,crossing_now,left_speed,right_speed,control_mode,"ROADWORK");
            tracking_mode="ROADWORK"; break;
          case State::WORKERS_AHEAD:
            runSlowMode(lines,error,crossing_now,left_speed,right_speed,control_mode,"WORKERS_AHEAD");
            tracking_mode="WORKERS_AHEAD"; break;
          default:
            left_speed=right_speed=0.0; break;
        }
        break;
    }

    applyTrafficSpeedMultiplier(left_speed, right_speed);
    publishWheels(left_speed, right_speed);

std::string debug_tracking_mode = tracking_mode + " | mask=" + line_mask_method_;

cv::Mat debug = makeDebugFrame(frame,mask,polygon,lines,target_x,error,
  debug_tracking_mode,direction,scan_y,band_y1,band_y2,
  left_speed,right_speed,crossing_now,crossing_rect,crossing_fill,
  zebra_now,zebra_rect,zebra_count);    writeVideo(debug);

    frame_count_++;
    auto t  = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(t - last_log_time_).count();
    if (dt >= 0.5) {
      last_log_time_ = t;
      RCLCPP_INFO(get_logger(),
        "state=%s lines=%zu cross=%d zebra=%d zcnt=%d mode=%s err=%.1f L=%.3f R=%.3f sign=%s tl=%s action=%s active=%s slow=%d",
        stateName().c_str(), lines.size(), crossing_now?1:0, zebra_now?1:0, zebra_count,
        tracking_mode.c_str(), error, left_speed, right_speed,
        current_sign_.c_str(), traffic_light_state_.c_str(),
        pending_intersection_action_.c_str(), active_intersection_action_.c_str(),
        slow_mode_active_?1:0);
    }
  }

  // ===========================================================================
  // Member variables
  // ===========================================================================
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_pub_, right_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sign_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  cv::VideoCapture cap_;
  cv::VideoWriter  writer_;
  bool video_failed_{false};

  // Camera
  bool        gst_{true};
  std::string device_{"/dev/video0"};
  int         source_{0}, width_{640}, height_{480}, fps_{30};
  double      control_hz_{15.0};
  bool        use_image_topic_{false};
  std::string image_topic_{"/camera/image_raw"};
  bool        enable_wheel_commands_{true};
  int         disabled_startup_stop_frames_{12};
  int         disabled_startup_stop_remaining_{0};
  std::mutex  frame_mutex_;
  cv::Mat     latest_frame_;
  bool        have_latest_frame_{false};

  // Calibration / undistortion
  bool enable_undistort_{true};
  int calib_width_{1280}, calib_height_{720};
  double undistort_alpha_{0.0};
  bool undistort_maps_ready_{false};
  cv::Size undistort_map_size_;
  cv::Mat undistort_map1_, undistort_map2_;

  // Recording
  std::string out_path_{"/home/puzzlebot/lane_drive_debug.avi"};
  double      video_fps_{15.0};
  bool        side_by_side_{true};
  double      seconds_{20.0};

  // Vision
  int    adaptive_block_{41}, adaptive_c_{7}, black_ceiling_{115};
  double scan_y_norm_{0.72};
  int    band_height_{18}, hough_threshold_{18}, hough_min_length_{45}, hough_max_gap_{12};
  double hough_min_abs_angle_deg_{20.0}, group_gap_{45.0};
  int    fallback_offset_px_{160}, deadband_px_{25}, max_lost_frames_{30};

  // Movement
  double max_speed_{0.45};
  double speed_straight_{0.25}, speed_soft_turn_{0.16}, speed_hard_turn_{0.10}, search_speed_{0.04};
  double kp_straight_{0.0007}, kp_soft_turn_{0.0020}, kp_hard_turn_{0.0035};
  double small_error_px_{30.0}, medium_error_px_{80.0};
  double turn_sign_{1.0}, left_motor_sign_{1.0}, right_motor_sign_{1.0};

  // Odom curve FSM
  bool        enable_odom_curves_{true};
  std::string odom_topic_{"/odom"};
  double      curve_radius_m_{0.25}, curve_angle_deg_{90.0};
  double      curve_linear_speed_{0.18}, wheel_base_m_{0.19};
  double      min_curve_wheel_speed_{0.12};
  double      curve_direction_override_error_px_{45.0};
  double      line_follow_correction_limit_{0.70};
  double      curve_trigger_error_px_{65.0};   // entra antes a curva
  int         curve_trigger_frames_{1};
  double      curve_cooldown_s_{1.0};
  double      curve_duration_s_{1.8};          // duración por tiempo (reemplaza dead reckoning)
  double      reacquire_speed_{0.08};
  int         reacquire_frames_needed_{5};
  double      reacquire_timeout_s_{3.0};

  // Crossing
  bool   enable_crossing_gate_{true};
  double crossing_roi_y1_{0.55}, crossing_roi_y2_{0.90};
  int    crossing_min_area_{4500}, crossing_min_width_px_{180}, crossing_min_height_px_{25};
  double crossing_min_fill_ratio_{0.22}, crossing_max_fill_ratio_{0.85}, crossing_horizontal_ratio_{2.5};
  int    crossing_clear_frames_needed_{4};
  bool   crossing_requires_three_lines_{true};

  // Center-lane-only mode
  bool   use_center_lane_only_{true};
  double center_lane_x_min_{0.34}, center_lane_x_max_{0.66};

  // Zebra stop ROI
  bool   enable_zebra_stop_roi_{true};
  double zebra_roi_x1_{0.03}, zebra_roi_x2_{0.97}, zebra_roi_y1_{0.60}, zebra_roi_y2_{0.78};
  int    zebra_min_blobs_{3}, zebra_frames_needed_{1};
  int    zebra_min_area_{250}, zebra_min_width_px_{35}, zebra_min_height_px_{6}, zebra_max_height_px_{45};
  double zebra_min_horizontal_ratio_{2.4}, zebra_min_fill_ratio_{0.25};
  bool   zebra_active_{false};
  int    zebra_seen_count_{0};

  // Stop-at-zebra behavior
  bool   stop_at_zebra_enabled_{true};
  bool   give_away_stops_at_zebra_{false};
  double stop_zebra_arm_timeout_s_{6.0};
  bool   pending_stop_at_zebra_{false};
  std::string pending_stop_reason_{"NONE"};
  rclcpp::Time pending_stop_start_time_{0, 0, RCL_ROS_TIME};

  // Sign parameters
  double sign_timeout_s_{3.0};
  double stop_duration_s_{2.0};
  double give_away_speed_factor_{0.5};
  double slow_speed_factor_{0.5};
  double traffic_green_speed_multiplier_{2.0};
  double traffic_yellow_speed_multiplier_{1.0};
  double traffic_light_timeout_s_{8.0};
  double intersection_action_timeout_s_{8.0};
  double intersection_entry_speed_{0.65};
  double intersection_entry_duration_s_{0.75};
  double intersection_forward_duration_s_{1.25};
  double intersection_spin_speed_{0.55};
  double intersection_spin_duration_s_{0.85};
  double forward_duration_s_{1.5};

  // FSM state
  State        state_{State::LINE_FOLLOW};
  rclcpp::Time state_start_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_curve_end_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time end_time_{0, 0, RCL_ROS_TIME};

  int    lost_counter_{0}, curve_trigger_count_{0}, reacquire_count_{0};
  bool   crossing_active_{false};
  int    crossing_clear_count_{0};
  std::string curve_direction_{"RIGHT"};
  std::string last_non_center_direction_{"RIGHT"};
  std::string line_mask_method_{"NONE"};

  // Odometry (para salida por distancia si está disponible)
  bool   have_odom_{false};
  double current_odom_x_{0.0}, current_odom_y_{0.0};
  double curve_start_x_{0.0}, curve_start_y_{0.0};
  double curve_target_arc_m_{0.0};     // distancia de arco objetivo
  double curve_radius_m_cur_{0.25};    // radio de la curva actual

  // Sign state
  std::string  current_sign_{"NONE"};
  rclcpp::Time last_sign_time_;
  std::string  traffic_light_state_{"NONE"};
  rclcpp::Time last_traffic_light_time_{0, 0, RCL_ROS_TIME};
  std::string  pending_intersection_action_{"NONE"};
  std::string  active_intersection_action_{"NONE"};
  rclcpp::Time last_intersection_action_time_{0, 0, RCL_ROS_TIME};

  // Speed management
  double normal_max_speed_{0.45};
  bool   slow_mode_active_{false};

  int    frame_count_{0};
  std::chrono::steady_clock::time_point last_log_time_{std::chrono::steady_clock::now()};
};

// =============================================================================
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<DirectCameraLaneFollowerRecorder>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
  }
  rclcpp::shutdown();
  return 0;
}
