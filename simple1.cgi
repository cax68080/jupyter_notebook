#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import sys
import codecs

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
#sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
print 'Content-type: text/plain charset=utf-8\n'
print ''
print 'こんにちは'