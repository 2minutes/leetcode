using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[AddComponentMenu("MyGame/Player")]
public class Player : MonoBehaviour
{
    public float m_speed = 10;
    protected Transform m_transform;
    public Transform m_ball;

    private Animator m_animator; // 调用animator

    // private ControlMode m_controlMode = ControlMode.Direct; // 和转向相关 枚举类中有Direct和Tank, Direct 表示上下左右相对于视角, Tank表示上下作用相对于自己,用Direct就可以

    private float m_currentV = 0; // 垂直加速度
    private float m_currentH = 0; // 水平加速度

    private readonly float m_interpolation = 10; //相当于加速度的加加速度的倒数，意思是经过10单位的时间可以从一个值变为另一个值

    private Vector3 m_currentDirection = Vector3.zero;
    bool hasBall = true;
    // Start is called before the first frame update

    // public void Initialize(GameObject character)
    // {
    //     m_animator = character.GetComponent<Animator>();
    // }
    // void Awake()
    // {
    //     if (!m_animator) { gameObject.GetComponent<Animator>(); }
    // }

    void Start()
    {
        m_transform = this.transform;
        m_animator = GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
        // float movev = 0;
        // float moveh = 0;

        // if (Input.GetKey(KeyCode.UpArrow))
        // {
        //     movev += m_speed * Time.deltaTime;
        // }

        // if (Input.GetKey(KeyCode.DownArrow))
        // {
        //     movev -= m_speed * Time.deltaTime;
        // }
        // if (Input.GetKey(KeyCode.LeftArrow))
        // {
        //     moveh -= m_speed * Time.deltaTime;
        // }
        // if (Input.GetKey(KeyCode.RightArrow))
        // {
        //     moveh += m_speed * Time.deltaTime;
        // }
        // this.m_transform.Translate(new Vector3(moveh, 0, movev));

        if (hasBall)
        {
            if (Input.GetKey(KeyCode.Space) || Input.GetMouseButton(0))
            {
                Instantiate(m_ball, m_transform.position, m_transform.rotation);
                hasBall = false;
            }
        }
    }

    //     MonoBehaviour.Update 更新
    // 当MonoBehaviour启用时，其Update在每一帧被调用。

    // MonoBehaviour.FixedUpdate 固定更新
    // 当MonoBehaviour启用时，其 FixedUpdate在每一帧被调用。

    // 处理Rigidbody时，需要用FixedUpdate代替Update。例如:给刚体加一个作用力时，你必须应用作用力在FixedUpdate里的固定帧，而不是Update中的帧。(两者帧长不同)
    void FixedUpdate()
    {

        DirectUpdate();
        if (Input.GetKey(KeyCode.Z)) 
        //小人按Z触发，从empty到death状态。写在了DemoAnimator的Animation Layer的状态图里。
        //TODO
        //目前只有从Empty到Death一种触发,后续要让小人爬起来，写一个触发Trigger，设置一种Death到Empty的状态转移，设置为该条件下触发，就可以爬起来了
        
        {
            m_animator.SetTrigger("Death");
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.tag.CompareTo("EnemyBall") == 0)
        {
            EnemyBall enemyBall = other.GetComponent<EnemyBall>();
            if (enemyBall != null)
            {
                // m_life-=ball.m_power;
                // if(m_life<=0){
                //     Destroy(this.gameObject);
                // }
                Destroy(this.gameObject);
            }
        }
    }

    private enum ControlMode
    {
        /// <summary>
        /// Up moves the character forward, left and right turn the character gradually and down moves the character backwards
        /// </summary>
        Tank,
        /// <summary>
        /// Character freely moves in the chosen direction from the perspective of the camera
        /// </summary>
        Direct
    }

    private void DirectUpdate()
    {
        // 1.Vertical                        对应键盘上面的上下箭头，当按下上或下箭头时触发
        // 2.Horizontal                    对应键盘上面的左右箭头，当按下左或右箭头时触发
        // 这个返回的是一个渐变的量，所以会引起步态的，所以短按是0.1 长按是1,相当于加速度
        float v = Input.GetAxis("Vertical");
        float h = Input.GetAxis("Horizontal");

        Transform camera = Camera.main.transform;

        // if (Input.GetKey(KeyCode.LeftShift))
        // {
        //     v *= m_walkScale;
        //     h *= m_walkScale;
        // }

        //这里是为了平顺 Lerp返回的是线性差值的数值，加速度分量从原始值变为新的值
        m_currentV = Mathf.Lerp(m_currentV, v, Time.deltaTime * m_interpolation);
        m_currentH = Mathf.Lerp(m_currentH, h, Time.deltaTime * m_interpolation);

        Vector3 direction = camera.forward * m_currentV + camera.right * m_currentH; //算出此时加速度的方向，也就是脸朝的方向


        float directionLength = direction.magnitude;
        direction.y = 0;
        direction = direction.normalized * directionLength;

        if (direction != Vector3.zero) //如果此时有加速度，才需要如下运算
        {
            m_currentDirection = Vector3.Slerp(m_currentDirection, direction, Time.deltaTime * m_interpolation); //差值平滑

            transform.rotation = Quaternion.LookRotation(m_currentDirection); //转体
            transform.position += m_currentDirection * m_speed * Time.deltaTime; //位置移动
            m_animator.SetFloat("MoveSpeed", direction.magnitude); //设置MoveSpeed给animator调整动作
        }
    }
}
