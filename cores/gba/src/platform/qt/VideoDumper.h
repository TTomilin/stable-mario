/* Copyright (c) 2013-2017 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#pragma once

#include <QAbstractVideoSurface>

namespace QGBA {

class VideoDumper : public QAbstractVideoSurface {
Q_OBJECT

public:
	VideoDumper(QObject* parent = nullptr);

	bool present(const QVideoFrame& frame) override;
	QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType type = QAbstractVideoBuffer::NoHandle) const override;

signals:
	void imageAvailable(const QImage& image);
};

}
