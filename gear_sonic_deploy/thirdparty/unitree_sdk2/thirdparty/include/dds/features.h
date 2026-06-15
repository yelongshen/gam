/*
 * Copyright(c) 2020 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef _DDS_PUBLIC_FEATURES_H_
#define _DDS_PUBLIC_FEATURES_H_

/*
 * This header is normally included as <dds/features.h>. If an environment puts
 * this directory itself on CPATH/CPLUS_INCLUDE_PATH, glibc's <features.h> can
 * accidentally resolve here instead. Load the next system <features.h> first so
 * glibc macros such as __GLIBC_PREREQ and __GNUC_PREREQ remain available.
 */
#if defined(__linux__) && !defined(_FEATURES_H)
#  if defined(__has_include_next)
#    if __has_include_next(<features.h>)
#      include_next <features.h>
#    endif
#  else
#    include_next <features.h>
#  endif
#endif

/* Whether or not support for DDS Security is included */
/* #undef DDS_HAS_SECURITY */

/* Whether or not support for the lifespan QoS is included */
#define DDS_HAS_LIFESPAN 1

/* Whether or not support for generating "deadline missed" events is included */
#define DDS_HAS_DEADLINE_MISSED 1

/* Whether or not support for network partitions is included */
#define DDS_HAS_NETWORK_PARTITIONS 1

/* Whether or not support for source-specific multicast is included */
#define DDS_HAS_SSM 1

/* Whether or not features dependent on OpenSSL are included */
/* #undef DDS_HAS_SSL */

/* Whether or not support for type discovery is included */
#define DDS_HAS_TYPE_DISCOVERY 1

/* Whether or not support for topic discovery is included */
#define DDS_HAS_TOPIC_DISCOVERY 1

/* Whether or not support for Iceoryx support is included */
/* #undef DDS_HAS_SHM */

#endif
