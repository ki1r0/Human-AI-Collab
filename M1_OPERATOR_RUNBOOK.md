# M1 操作员运行手册

本手册只适用于 `fault_hil`，不是用预录命令代替真人的脚本。

1. 操作员先确认 Isaac 窗口/流媒体连接中能看到 R1、红色
   `Hub_Cover_Output_Top` 和固定的 `Casing_Top`。检查 run 的
   `manifest.json`、episode ID 和 fault seed。
2. 机器人失败后必须先进入 `SAFE_HOLD`。操作员点击/发送真实的
   `TAKE_CONTROL`，确认当前夹爪仍夹住端盖，不能拖动 viewport 中的 USD
   prim，也不能直接把端盖放入 socket。
3. 仅使用 broker 提供的受限笛卡尔 jog；每个平移轴命令不超过 10 mm。
   调整端盖的对准/倾角，保持它尚未坐合。每条命令、反馈位姿和时间戳
   都会写进公开事件流。
4. 调整完成后发送 `RETURN_CONTROL_AND_DONE`。确认控制权显示为 AUTO，
   并等待一帧全新的 observation；旧 observation 或旧插装轨迹不能重放。
5. 后续插入、释放、撤爪和静置验证必须由机器人自主完成。若通信、夹持、
   安全边界或可见性异常，立即 SAFE_STOP 并中止该回合。

没有真人在线时，runner 必须保留 `WAITING_FOR_OPERATOR`，不可用固定脚本、
预录 teleop 或 evaluator 标签自动通过 HIL。
