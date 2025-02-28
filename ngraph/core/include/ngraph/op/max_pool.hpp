// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "ngraph/op/util/max_pool_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Batched max pooling operation.
            class NGRAPH_API MaxPool : public op::util::MaxPoolBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs a batched max pooling operation.
                MaxPool() = default;

                /// \brief Constructs a batched max pooling operation.
                ///
                /// \param arg The node producing the input data batch tensor.
                /// \param strides The strides.
                /// \param pads_begin The beginning of padding shape.
                /// \param pads_end The end of padding shape.
                /// \param kernel The kernel shape.
                /// \param rounding_type Whether to use ceiling or floor rounding type while
                /// computing output shape.
                /// \param auto_pad The pad type for automatically computing padding sizes.
                MaxPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        const op::RoundingType rounding_type = op::RoundingType::FLOOR,
                        const PadType auto_pad = op::PadType::EXPLICIT);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The default value for MaxPool.
                NGRAPH_SUPPRESS_DEPRECATED_START
                virtual std::shared_ptr<Node> get_default_value() const override;
                NGRAPH_SUPPRESS_DEPRECATED_END

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;
                bool has_evaluate() const override;

            private:
                bool evaluate_maxpool(const HostTensorVector& outputs,
                                      const HostTensorVector& inputs) const;
            };
        } // namespace v1

        namespace v8
        {
            /// \brief MaxPooling operation with values and indices calculated as individual outputs
            class NGRAPH_API MaxPool : public op::util::MaxPoolBase
            {
            public:
                NGRAPH_RTTI_DECLARATION;

                /// \brief Constructs an empty MaxPool operation.
                MaxPool() = default;

                /// \brief Constructs a parametrized MaxPool operation.
                ///
                /// \param arg Output of a node producing the feature tensor to be pooled.
                /// \param strides The strides of the pooling filter.
                /// \param dilations The dilations of the pooling filter.
                /// \param pads_begin Paddings at the beginning of each spatial axis.
                /// \param pads_end Paddings at the end of each spatial axis.
                /// \param kernel The kernel shape.
                /// \param rounding_type Whether to use ceiling or floor rounding type while
                ///                      computing the output shape.
                /// \param auto_pad The pad type for automatic calculation of the padding sizes.
                /// \param index_element_type The data type used by the second output tensor
                ///                           containing the selected indices.
                /// \param axis Indicates a dimension in the input data shape which should be used
                ///             as a starting point for calculation of the upper bound of allowed
                ///             values of the indices output.
                MaxPool(const Output<Node>& arg,
                        const Strides& strides,
                        const Strides& dilations,
                        const Shape& pads_begin,
                        const Shape& pads_end,
                        const Shape& kernel,
                        const op::RoundingType rounding_type = op::RoundingType::FLOOR,
                        const PadType auto_pad = op::PadType::EXPLICIT,
                        const element::Type index_element_type = element::i64,
                        const int64_t axis = 0,
                        const float pads_value = -std::numeric_limits<float>::infinity());

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The pooling filter's dilations.
                const Strides& get_dilations() const noexcept { return m_dilations; }
                void set_dilations(const Strides& dilations) { m_dilations = dilations; }

                /// \return The data type of the second output tensor (indices).
                element::Type get_index_element_type() const noexcept
                {
                    return m_index_element_type;
                }
                void set_index_element_type(const element::Type index_element_type)
                {
                    m_index_element_type = index_element_type;
                }

                // \return The 'axis' attribute value.
                int64_t get_axis() const { return m_axis; }
                void set_axis(const int64_t axis) { m_axis = axis; }

                // \return The value stored in the padding cells.
                float get_pads_value() const { return m_pads_value; }
                void set_pads_value(const float pads_value) { m_pads_value = pads_value; }

            private:
                Strides m_dilations;
                element::Type m_index_element_type{element::i32};
                int64_t m_axis{0};
                float m_pads_value{-std::numeric_limits<float>::infinity()};
            };
        } // namespace v8
    }     // namespace op
} // namespace ngraph
