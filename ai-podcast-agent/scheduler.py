"""
定时调度器

使用 APScheduler 实现定时任务调度
每天 09:00 自动运行播客生成流程
"""

import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


class PodcastScheduler:
    """播客定时调度器"""

    def __init__(self, config, task_func):
        """
        初始化调度器

        Args:
            config: 配置对象
            task_func: 要执行的任务函数
        """
        self.config = config
        self.task_func = task_func

        self.scheduler_config = config.SCHEDULER_CONFIG
        self.timezone = pytz.timezone(self.scheduler_config["timezone"])
        self.daily_run_time = self.scheduler_config["daily_run_time"]
        self.enable_scheduler = self.scheduler_config["enable_scheduler"]
        self.run_on_startup = self.scheduler_config["run_on_startup"]

        # 创建调度器
        self.scheduler = BlockingScheduler(timezone=self.timezone)

    def start(self):
        """启动调度器"""
        if not self.enable_scheduler:
            logger.warning("调度器未启用，仅运行一次任务")
            self._run_task()
            return

        logger.info(f"正在配置定时调度器...")
        logger.info(f"时区：{self.timezone}")
        logger.info(f"每日运行时间：{self.daily_run_time}")

        # 解析运行时间
        hour, minute = map(int, self.daily_run_time.split(':'))

        # 添加定时任务
        self.scheduler.add_job(
            self._run_task,
            trigger=CronTrigger(hour=hour, minute=minute, timezone=self.timezone),
            id='daily_podcast_generation',
            name='每日播客生成',
            replace_existing=True
        )

        # 如果配置了启动时运行
        if self.run_on_startup:
            logger.info("启动时立即运行一次任务...")
            self._run_task()

        # 显示下次运行时间
        next_run = self.scheduler.get_jobs()[0].next_run_time
        logger.info(f"下次运行时间：{next_run}")

        # 启动调度器
        logger.info("调度器已启动，等待执行...")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("调度器已停止")
            self.scheduler.shutdown()

    def _run_task(self):
        """
        执行任务

        捕获异常以防止调度器崩溃
        """
        try:
            logger.info("=" * 60)
            logger.info(f"开始执行任务 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)

            # 执行任务函数
            self.task_func()

            logger.info("=" * 60)
            logger.info("任务执行完成")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)

            # 发送错误通知（如果配置了）
            self._send_error_notification(e)

    def _send_error_notification(self, error: Exception):
        """
        发送错误通知

        Args:
            error: 异常对象
        """
        notification_config = self.config.NOTIFICATION_CONFIG

        if not notification_config["enable_notification"]:
            return

        # TODO: 实现通知发送
        logger.warning("错误通知功能待实现")

    def run_once(self):
        """立即运行一次任务"""
        logger.info("手动触发任务执行...")
        self._run_task()

    def get_next_run_time(self):
        """获取下次运行时间"""
        if not self.enable_scheduler or not self.scheduler.get_jobs():
            return None

        return self.scheduler.get_jobs()[0].next_run_time

    def stop(self):
        """停止调度器"""
        if self.scheduler.running:
            logger.info("正在停止调度器...")
            self.scheduler.shutdown()
            logger.info("调度器已停止")


def main():
    """测试函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 导入配置
    import config

    # 定义测试任务
    def test_task():
        logger.info("这是一个测试任务")
        logger.info(f"当前时间：{datetime.now()}")

    # 创建调度器
    scheduler = PodcastScheduler(config, test_task)

    # 显示下次运行时间
    logger.info(f"配置的运行时间：{scheduler.daily_run_time}")

    # 运行一次测试
    scheduler.run_once()


if __name__ == "__main__":
    main()
