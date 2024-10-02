# SPDX-FileCopyrightText: 2024-present Sunil Thaha <sthaha@redhat.com>
#
# SPDX-License-Identifier: MIT
import sys

if __name__ == "__main__":
    from power_model.cli import power_model

    sys.exit(power_model())
