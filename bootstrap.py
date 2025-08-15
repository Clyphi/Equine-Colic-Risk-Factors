#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project: Equine-Colic-Risk-Factors
File: bootstrap
Author: Claudia Leins
Description: 
Bootstrap-Modul: Stellt sicher, dass das Projekt-Root im sys.path liegt
und bietet eine PROJECT_ROOT-Variable für konsistente Pfade.
"""

import sys
from pathlib import Path

# Projekt-Root bestimmen (Ordner, in dem bootstrap.py liegt)
PROJECT_ROOT = Path(__file__).resolve().parent

# Projekt-Root in sys.path einfügen, falls noch nicht vorhanden
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
