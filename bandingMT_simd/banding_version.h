﻿//  -----------------------------------------------------------------------------------------
//    バンディング低減MT SIMD by rigaya
//  -----------------------------------------------------------------------------------------
//   ソースコードについて
//   ・無保証です。
//   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
//   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
//  -----------------------------------------------------------------------------------------


#pragma once
#ifndef _BANDING_VER_H_
#define _BANDING_VER_H_

#define AUF_VERSION      0,0,17,5
#define AUF_VERSION_STR  "17+5"
#define AUF_NAME         "bandingMT_simd.auf"
#define AUF_FULL_NAME    "バンディング低減MT SIMD"
#define AUF_VERSION_NAME "バンディング低減MT SIMD ver17+4"
#define AUF_VERSION_INFO AUF_VERSION_NAME

#ifdef DEBUG
#define VER_DEBUG   VS_FF_DEBUG
#define VER_PRIVATE VS_FF_PRIVATEBUILD
#else
#define VER_DEBUG   0
#define VER_PRIVATE 0
#endif

#define VER_STR_COMMENTS         AUF_FULL_NAME
#define VER_STR_COMPANYNAME      ""
#define VER_STR_FILEDESCRIPTION  AUF_FULL_NAME
#define VER_FILEVERSION          AUF_VERSION
#define VER_STR_FILEVERSION      AUF_VERSION_STR
#define VER_STR_INTERNALNAME     AUF_FULL_NAME
#define VER_STR_ORIGINALFILENAME AUF_NAME
#define VER_STR_LEGALCOPYRIGHT   AUF_FULL_NAME
#define VER_STR_PRODUCTNAME      "bandingMT SIMD"
#define VER_PRODUCTVERSION       VER_FILEVERSION
#define VER_STR_PRODUCTVERSION   VER_STR_FILEVERSION

#endif //_BANDING_VER_H_
