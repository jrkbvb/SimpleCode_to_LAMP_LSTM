#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:25:25 2021

@author: samedwards
"""
def padn(d,n):
    try:
        c = '{0:0>' + str(n) + '}'
        return c.format(int(d))
    except:
        return d