/* Copyright (c) 2013-2017 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#include "ColorPicker.h"

#include <QColorDialog>
#include <QEvent>

using namespace QGBA;

ColorPicker::ColorPicker() {
}

ColorPicker::ColorPicker(QWidget* parent, const QColor& defaultColor)
	: m_parent(parent)
	, m_defaultColor(defaultColor)
{
	QPalette palette = parent->palette();
	palette.setColor(parent->backgroundRole(), defaultColor);
	parent->setPalette(palette);
	parent->installEventFilter(this);
}

ColorPicker& ColorPicker::operator=(const ColorPicker& other) {
	if (m_parent) {
		m_parent->removeEventFilter(this);
	}
	m_parent = other.m_parent;
	m_defaultColor = other.m_defaultColor;
	m_parent->installEventFilter(this);

	return *this;
}

bool ColorPicker::eventFilter(QObject* obj, QEvent* event) {
	if (event->type() != QEvent::MouseButtonRelease) {
		return false;
	}
	int colorId;
	if (obj != m_parent) {
		return false;
	}

	QWidget* swatch = static_cast<QWidget*>(obj);

	QColorDialog* colorPicker = new QColorDialog;
	colorPicker->setAttribute(Qt::WA_DeleteOnClose);
	colorPicker->setCurrentColor(m_defaultColor);
	colorPicker->open();
	connect(colorPicker, &QColorDialog::colorSelected, [this, swatch](const QColor& color) {
		m_defaultColor = color;
		QPalette palette = swatch->palette();
		palette.setColor(swatch->backgroundRole(), color);
		swatch->setPalette(palette);
		emit colorChanged(color);
	});
	return true;
}
