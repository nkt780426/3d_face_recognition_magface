# BRDF: Bidirectional Reflectance Distribution Function

# Mục tiêu và vấn đề

Ở bài 1 và 2, ta đã biết được surface radiance (độ sáng của bề mặt/đơn vị diện tích/đơn vị góc) bị ảnh hưởng bởi các thuộc tính liên quan đến chất liệu vật thể hay reflectance properties (các đặc tính phản xạ). Vật liệu nào cũng có khả năng hấp thụ năng lượng của ánh sáng phát ra từ nguồn và chuyển thành nhiệt năng 1 phần sau đó mới tán xạ thành các tia sáng (tuy nhiên lượng năng lượng này thường được coi là rất nhỏ và bị bỏ qua). Chủ yếu là tần xuất các tia phản xạ hay quang phổ của mỗi vật liệu là khác nhau.

Trên đời có rất nhiều các loại vật liệu và tính chất của mỗi loại có thể là tổng hợp của rất nhiều tính chất khác. Chúng ta muốn tìm ra cách ngắn gọn để biểu diễn các reflectance properties  (các đặc tính phản xạ) của mọi vật thể => BRDF (hàm phân phối phản xạ 2 chiều)

1. Surface Appearance (hình dạng vật thể)
    ![surface appearance](images/4.%20Surface%20appearance.png)

    Có 1 loạt các quả cầu được chiếu sáng bởi nguồn sáng rất xa và phức tạp, có thể coi các quả cầu đều được chiếu sáng như nhau. Camera cũng được đặt rất xa có thể coi direction của các quả cầu đến camera là như nhau.

    Lý do duy nhất chúng ta thấy được các quả cầu khác nhau là do vật liệu của chúng khác nhau dẫn đến khả năng reflectance khác nhau.

2. Surface Reflectance (khả năng phản xạ của vật thể)

    ![Surface reflectance](images/4.%20Surface%20refectance.png)

    Như các bài trước đã biết, khả năng phản xạ của vật thể phụ thuộc vào hướng chiếu sáng của nguồn (illumination) và hướng nhìn của camera (view) hay hướng của 1 tia tán xạ và chúng ta muốn mô tả mối quan hệ giữa chúng.

3. Bidirectional reflectance distribution function (mô tả 2 hướng trên)

    ![BRDF](images/4.%20BRDF.png)

    Để mô tả 1 tia trong Oxyz cần phải xác định điểm gốc của tia và 2 góc
        Theta là góc zenith(thiên đỉnh) là góc tạo với tia Oz
        Phi là góc azimuth(phương vị) là góc ở mặt phẳng Oxy
    Tận dụng điều trên ta có thể mô tả hướng chiếu sáng của nguồn đến bề mặt và tia phản xạ của nó theo 2 góc trên.
        E(phi i, theta i): Irradiance due to source (khả năng chiếu sáng của nguồn) theo hướng (phi i, theta i)
        L(phi i, theta i): Radiance of surface (khả năng phản xạ của bề mặt) theo hướng (phi i, theta i)
    Từ đó ta rút ra được hàm BRDF như trên hình (đơn vị là 1/sr) là hàm 4 chiều. Hàm này mô tả đầy đủ đặc tính phản xạ của bất kỳ điểm nào trong scene.

4. Các tính chất của BRDF.

    ![BRDF properties](images/4.%20BRDF%20properties%201.png)
    
    Tính chất 1: luôn lớn hơn 0.
    Tính chất 2: Helmholtz Reciprocity (độ tương hỗ Helmholtz) nó nói về khả năng đảo ngược tia chiếu sáng và tia phản xạ thì sẽ nhận được cùng BRDF

5. BRDF of isotropic surfaces

    ![BRDF isotropic](images/4.%20BRDF%20isotropic.png)

    Mặc dù là hàm 4 chiều nhưng trong những trường hợp đặc biệt. Nó có thể được mô tả làm hàm 3 chiều
    Trường hợp 3 chiều là Isotropic surface (bề mặt đẳng hướng, nơi tia chiếu sáng đối xứng tia phản xạ). BRDF của nó như hình
    Cách dễ nhất để biết 1 bề mặt là quay, đối xứng hay đẳng hướng là khi bạn quan sát bề mặt vật thể và biết vector pháp tuyến bề mặt tại điểm đó, bạn quay vật thể sao cho khi quay điểm đó trong mắt bạn ở nguyên vị trí đó và không thay đổi thì có thể nói nó đối xứng quay về mặt phản xạ hay đẳng hướng (isotropic)
    Nói rõ hơn ở phần 6

6.  BRDF isotropic(đẳng hướng) and BRDF anisotropic(dị hướng).

    ![Isotropic and anisotropic](images/4.%20Isotropic%20and%20Anisotropic.png)

    Nếu bạn đặt quả cầu 1 này trong cùng điều kiện, bạn sẽ không di chuyển vị trí của nó mà chỉ xoay quả cầu. Image của quả cầu sẽ không thay đổi.

    BRDF (Bidirectional Reflectance Distribution Function) là một hàm dùng để mô tả cách ánh sáng phản xạ từ một bề mặt. Nó biểu diễn mối quan hệ giữa ánh sáng tới và ánh sáng phản xạ từ bề mặt tại một điểm, phụ thuộc vào hướng của nguồn sáng và hướng quan sát.
    
    BRDF được gọi là isotropic (đẳng hướng) khi phản xạ ánh sáng không phụ thuộc vào hướng quay của bề mặt xung quanh trục pháp tuyến (trục vuông góc với bề mặt).
        Đặc điểm: Phản xạ của bề mặt là đồng đều theo mọi hướng quay của bề mặt quanh pháp tuyến. Điều này có nghĩa là sự phản xạ ánh sáng chỉ phụ thuộc vào hướng tới và hướng quan sát, mà không phụ thuộc vào cách quay của bề mặt.
        Ví dụ: Các bề mặt nhẵn, không có cấu trúc chi tiết nhỏ theo hướng cố định như gương phẳng, bề mặt mịn hoặc những vật liệu có tính chất đồng nhất (ví dụ: nhựa, kim loại bóng mịn).
    
    BRDF được gọi là anisotropic (dị hướng) khi phản xạ ánh sáng phụ thuộc vào hướng quay của bề mặt xung quanh trục pháp tuyến.
        Đặc điểm: Phản xạ của bề mặt thay đổi khi quay bề mặt quanh trục pháp tuyến. Điều này có nghĩa là sự phản xạ không chỉ phụ thuộc vào hướng tới và hướng quan sát, mà còn phụ thuộc vào cách mà bề mặt bị xoay.
        Ví dụ: Các bề mặt có cấu trúc không đồng đều, ví dụ như tóc, vải lụa, hoặc bề mặt có các đường rãnh nhỏ theo hướng cố định (kim loại chải xước), nơi mà sự phản xạ ánh sáng thay đổi tùy thuộc vào góc quay của bề mặt.