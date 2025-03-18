# 代码架构
主函数：`control_robot()`

机器人参数加载：

    from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
    def func(robot: Robot, cfg: cfg)

    [lerobot.common.robot_devices.robots.utils] from lerobot.common.robot_devices.robots.configs import (robot_cfg)


### control_robot() 

`init_logging()`配置了日志系统，并定制了日志输出的格式。
- 日志格式：每条日志会输出日志级别、时间、文件路径（文件名 + 行号）以及日志消息内容。
- 日志级别：设置为 INFO，意味着只会处理 INFO 及更高级别的日志。
- 日志输出方式：将日志输出到控制台，并应用自定义的格式。

`robot = make_robot_from_config(cfg.robot)`实例化机器人类

判断机器人控制类型：

    # 参考代码顶部说明，分别对应说明中的几种任务
    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record(robot, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi
        run_lekiwi(cfg.robot)

    # 控制循环结束之后断连，可能需要人为断连（相机等进程不一定会停止）
    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()

## 控制部分函数介绍
### calibrate(robot: Robot, cfg: CalibrateControlConfig)

检查机器人的类型，如果是以stretch开头且未连接，则连接，如果没有初始化，则初始化

    if robot.robot_type.startswith("stretch"):
            if not robot.is_connected:
                robot.connect()
            if not robot.is_homed():
                robot.home()
            return

## 机器人参数定义(lerobot.common.robot_devices.robots.utils) 
- get_arm_id(name, arm_type)

会返回机器人手臂的类型，例如：`left_follower, right_follower, left_leader, or right_leader`

- class Robot(Protocol)

包含底层接口：

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...

- make_robot_config(robot_type: str, **kwargs) -> RobotConfig:

返回机器人对应的cfg。

- make_robot(robot_type: str, **kwargs) -> Robot:

实例化cfg，然后传递给`make_robot_from_config(config: RobotConfig)`创建机器人类

- make_robot_from_config(config: RobotConfig)

根据机器人的类别进行分类，根据提供的cfg创建对应的机器人类。

    # 机械臂机器人（固定？）
    if isinstance(config, ManipulatorRobotConfig):
        from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
        return ManipulatorRobot(config)
    # 移动的机械臂机器人
    elif isinstance(config, LeKiwiRobotConfig):
        from lerobot.common.robot_devices.robots.mobile_manipulator import MobileManipulator
        return MobileManipulator(config)
    # 伸缩机械臂机器人
    else:
        from lerobot.common.robot_devices.robots.stretch import StretchRobot
        return StretchRobot(config)

## 机器人实例化（以ManipulatorRobot为例）
- def ensure_safe_goal_position(goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]):

检测当前的pos和目标的pos差异，min（最大相对距离，当前差异）避免当前位置和目标位置距离过远；max（-最大相对距离， 安全距离）避免距离过近？

### class ManipulatorRobot
我认为：follower -> 机械臂， leader -> 摇杆

cfg参数更新方式，相机参数更新也是类似的：

    leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config) # 实例化

**在进行每一次机械臂操作前都会判断机械臂是否连接**

- robot.connect()

连接机器人；对机器人进行校准；设置机器人到预备状态（初始位姿？）；扭曲激活；夹爪启动；检测是否可以获取位姿；激活相机

    # 连接
    for name in self.follower_arms:
        pollower_arms[name].connect()
    # 关节激活？
    if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
    # 扭矩激活
    for name in self.follower_arms:
        self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
    for name in self.follower_arms:
        print(f"Activating torque on {name} follower arm.")
        self.follower_arms[name].write("Torque_Enable", 1)



- \_\_del\_\_(self) & disconnect(self)

断开各机械臂以及相机的连接

-  send_action(self, action: torch.Tensor) -> torch.Tensor

由于传送给机械臂的action一定需要经过安全检测，所以action一定是在安全区间内的action；将获取到的action传送给机械臂对应arm的对应属性：`self.follower_arms[name].write("Goal_Position", goal_pos)`.

    # 将arm编码，方便后续根据编码赋值
    for name in self.follower_arms:
        # Get goal position of each follower arm by splitting the action vector
        to_idx += len(self.follower_arms[name].motor_names)
        goal_pos = action[from_idx:to_idx]
        from_idx = to_idx

    # read来获取当前的action，然后计算安全区域来剔除actions中超出范围的action
    present_pos = self.follower_arms[name].read("Present_Position")

    # 拼接 
    action_sent.append(goal_pos)
    return torch.cat(action_sent)

- capture_observation(self)
1. 获取follower arms的positon，计时
2. 将position转化为state
3. 获取cameras的图像（异步） 计时
4. 将state与image整合为obs_dict

- teleop_step( self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
1. 获取leader pos
2. leader pos对应与follower来说就是目标pos 与传送action的内容大致一样，要经过安全检测
3. 获取follower的pos
4. 整合follower当前的pos和目标pos
5. 获取相机的信息
6. 将follower当前的pos和相机image整合为obs_dict, goal_pos整合为action_dict

机械臂进入准备状态：设置PID参数为默认值或者小于默认值；解锁

    set_aloha_robot_preset(self)
    set_so100_robot_preset(self)
    set_koch_robot_preset(self)

- activate_calibration(self)

    - load_or_run_calibration_(name, arm, arm_type)

    加载校准文件，如果没有，则根据机械臂的型号等后加载对应的校准参数，可能需要手动校准

递归调用校准函数对follower arms和leader arms进行校准

    for name, arm in self.follower_arms.items():
        calibration = load_or_run_calibration_(name, arm, "follower")
        arm.set_calibration(calibration)
    for name, arm in self.leader_arms.items():
        calibration = load_or_run_calibration_(name, arm, "leader")
        arm.set_calibration(calibration)





    



