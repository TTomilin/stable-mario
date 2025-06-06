/* Copyright (c) 2013-2014 Jeffrey Pfau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
#pragma once

#include <QDialog>

#include "ColorPicker.h"

#include <mgba/core/core.h>

#ifdef M_CORE_GB
#include <mgba/gb/interface.h>
#endif

#include "ui_SettingsView.h"

namespace QGBA {

class ConfigController;
class InputController;
class ShortcutController;
class ShaderSelector;

class SettingsView : public QDialog {
Q_OBJECT

public:
	SettingsView(ConfigController* controller, InputController* inputController, ShortcutController* shortcutController, QWidget* parent = nullptr);
	~SettingsView();

	void setShaderSelector(ShaderSelector* shaderSelector);

signals:
	void biosLoaded(int platform, const QString&);
	void audioDriverChanged();
	void displayDriverChanged();
	void cameraDriverChanged();
	void pathsChanged();
	void languageChanged();
	void libraryCleared();

private slots:
	void selectBios(QLineEdit*);
	void updateConfig();
	void reloadConfig();

private:
	Ui::SettingsView m_ui;

	ConfigController* m_controller;
	InputController* m_input;
	ShaderSelector* m_shader = nullptr;

#ifdef M_CORE_GB
	uint32_t m_gbColors[12]{};
	ColorPicker m_colorPickers[12];
	static QList<enum GBModel> s_gbModelList;
#endif

	void saveSetting(const char* key, const QAbstractButton*);
	void saveSetting(const char* key, const QComboBox*);
	void saveSetting(const char* key, const QDoubleSpinBox*);
	void saveSetting(const char* key, const QLineEdit*);
	void saveSetting(const char* key, const QSlider*);
	void saveSetting(const char* key, const QSpinBox*);
	void saveSetting(const char* key, const QVariant&);

	void loadSetting(const char* key, QAbstractButton*, bool defaultVal = false);
	void loadSetting(const char* key, QComboBox*);
	void loadSetting(const char* key, QDoubleSpinBox*);
	void loadSetting(const char* key, QLineEdit*);
	void loadSetting(const char* key, QSlider*);
	void loadSetting(const char* key, QSpinBox*);
	QString loadSetting(const char* key);
};

}
