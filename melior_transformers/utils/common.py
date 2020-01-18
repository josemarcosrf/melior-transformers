import os
import logging

from melior_transformers.constants import ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL


def configure_colored_logging(loglevel=None):
    # more info on coloredlogs formatting:
    # https://coloredlogs.readthedocs.io/en/latest/api.html#changing-the-colors-styles
    import coloredlogs

    loglevel = loglevel or os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles["asctime"] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    # level_styles["debug"] = {}
    level_styles["debug"] = {"color": "white", "faint": True}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        # fmt="%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
        fmt="%(levelname)-8s %(name)s:%(lineno)s  - %(message)s",
        level_styles=level_styles,
        field_styles=field_styles,
    )
