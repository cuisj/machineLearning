#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  learn.py
#
#  Copyright 2017 崔士杰 <cuisj@asiainfo-sec.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import show

x = np.arange(-10, 10, 0.1)
y = 1 - (1.0 / (1.0 + np.exp(-x)))
y2 = - (1.0 / (1.0 + np.exp(-x)))

plot(x, y + y2)
show()
